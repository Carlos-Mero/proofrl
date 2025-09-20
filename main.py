import argparse
import json
import asyncio
from pathlib import Path
import random
import re
import logging
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
from trl import GRPOConfig, GRPOTrainer, TrlParser, ScriptArguments, ModelConfig
from trl.rewards import think_format_reward
from datasets import load_dataset, Dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from litellm import acompletion

from utils.async_runner import AsyncLoopThread

ASYNC_LOOP = AsyncLoopThread()

def extract_xml_content(text: str, tag: str):
    flags = re.DOTALL | 0
    pattern = rf"<{re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{re.escape(tag)}\s*>"

    last_content = None
    for m in re.finditer(pattern, text, flags):
        last_content = m.group(1)

    if last_content is None:
        return None
    return last_content.strip()

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def strip_think_simple(s: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

def get_current_log_path(log_dir: str):
    ts = datetime.now(timezone.utc).strftime("%m%dT%H%M")
    logdir = Path(log_dir) / ts
    return logdir

def prepare_dataset(dataset_path):
    """
    this function prepares datasets according to the given path
    """
    logger = logging.getLogger("dataset")
    logger.info("preparing dataset at path: %s", dataset_path)
    if dataset_path == "nproof/train.json":
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            problems = json.load(f)
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path == "nproof/valid.json":
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            problems = json.load(f)
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path == "HuggingFaceH4/MATH-500":
        ds = load_dataset(dataset_path)
        ds = ds.remove_columns(["solution"])
        ds = ds["test"]
    else:
        raise NotImplementedError(f"Unknown dataset name or path: {dataset_path}")

    # SYSTEM_PROMPT = (
    #     "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
    #     "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    #     "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
    #     "reasoning.\n</think>\nThis is my answer."
    # )
    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    )

    def make_conversations(example):
        return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example if isinstance(example, str) else example['problem']},
            ]
    prompts = [make_conversations(e) for e in ds]
    ds = ds.add_column(name="prompt", column=prompts)
    logger.info("completed preparing dataset at: %s", dataset_path)

    return ds

def accuracy_reward(completions, answer: list[str], **kwargs):
    """
    Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, answer):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {content}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(content.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards

def ttrl_reward(prompts, completions, **kwargs):
    # This function divides the completions into groups according to the prompts
    # So that it could be more robust for ttrl verification
    contents = [completion[0]["content"] for completion in completions]
    pairs = zip(prompts, contents)
    rewards = []
    for p, group in groupby(pairs, key=itemgetter(0)):
        comps = [c for _, c in group]
        parsed_anss = [parse(
            comp,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
            for comp in comps
        ]
        # The results of TTRL is exactly the row with maximum reward sum
        rewardsmetrix = [
            [float(verify(a, b)) for a in parsed_anss]
            for b in parsed_anss
        ]
        max_row = max(rewardsmetrix, key=sum)
        rewards += rewardsmetrix[max_row]
    return rewards

class LLMClient():
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    async def _infer_one(self,
                         messages,
                         sem: asyncio.Semaphore,
                         **kwargs) -> str:
        backoff = 1.0
        while True:
            try:
                async with sem:
                    resp = await acompletion(
                        model="openai/"+self.model,
                        messages=messages,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        drop_params=True,
                        **kwargs)
                return resp.choices[0].message["content"]
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ["rate", "timeout", "overloaded", "temporarily"]):
                    await asyncio.sleep(backoff + random.random() * 0.2)
                    backoff = min(backoff * 2, 60)
                    continue
                raise

    async def infer_batch_async(self,
                                all_messages,
                                concurrency: int = 8,
                                **kwargs) -> list[str]:
        logger = logging.getLogger("evaluator")
        logger.info(f"running batch inference on {len(all_messages)} samples")
        sem = asyncio.Semaphore(concurrency)
        ALLOWED_PARAM_KEYS = {"reasoning_effort"}
        infer_params = {k: v for k, v in kwargs.items() if k in ALLOWED_PARAM_KEYS}
        tasks = [
            asyncio.create_task(self._infer_one(messages, sem, **infer_params))
            for messages in all_messages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                raise RuntimeError(f"Task {i} failed") from r
        logger.info(f"completed batch inference on {len(all_messages)} samples")
        return results

class ProofRLEvaluator():
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    def verify(self, prompts, completions, **kwargs):
        all_messages = [
            [
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n"
                    "\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n"
                    "\n"
                    "If all of the above are correct, append `<verification>true</verification>` at the end of your reply; otherwise, append `<verification>false</verification>`.\n"
                    "\n"
                    f"<problem>{p[1]['content']}</problem>\n"
                    "\n"
                    f"<answer>{c if isinstance(c, str) else c[0]['content']}</answer>"
                )}
            ]
            for (p, c) in zip(prompts, completions)
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        rewards = [1.0 if extract_xml_content(r, "verification") == "true" else 0.0 for r in results]
        return rewards, results

    def __call__(self, prompts, completions, **kwargs):
        rewards, _ = self.verify(prompts, completions, **kwargs)
        return rewards

    @property
    def __name__(self):
        return "ProofRLEvaluator"

class ProofRLProver():
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    def __call__(self, problems: list[str], **kwargs):
        SYSTEM_PROMPT = (
            "You are an expert in math and is skilled in solving math problems."
        )
        all_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
            for problem in problems
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        return results

def train(script_args, grpo_cfg, model_cfg, custom_args):
    tdataset = prepare_dataset(custom_args.dataset)
    if custom_args.method == "rlvr":
        reward_funcs = [think_format_reward, accuracy_reward]
    elif custom_args.method == "ttrl":
        reward_funcs = [think_format_reward, ttrl_reward]
    elif custom_args.method == "proofrl":
        evaluator = ProofRLEvaluator(custom_args.eval_base_url, custom_args.api_key, custom_args.eval_model)
        reward_funcs = [think_format_reward, evaluator]
    else:
        raise NotImplementedError("Unknown training method")

    trainer = GRPOTrainer(
        model=model_cfg.model_name_or_path,
        args=grpo_cfg,
        train_dataset=tdataset,
        eval_dataset=None,
        reward_funcs=reward_funcs,
    )
    trainer.train()
    trainer.save_model(grpo_cfg.output_dir)

def eval(ns):
    logger = logging.getLogger("eval")
    logger.info("start verifying with proof_model: %s", ns.proof_model)
    logger.info("using eval model: %s", ns.eval_model)

    ds = prepare_dataset(ns.eval_dataset)
    prompts = [e['prompt'] for e in ds]
    problems = [e['problem'] for e in ds]
    prover = ProofRLProver(ns.prover_base_url, ns.api_key, ns.proof_model)
    if ns.method == "rlvr":
        evaluator = accuracy_reward
        answers = [e['answer'] for e in ds]
    elif ns.method == "ttrl":
        evaluator = ttrl_reward
    elif ns.method == "proofrl":
        evaluator = ProofRLEvaluator(ns.eval_base_url, ns.api_key, ns.eval_model)
    else:
        raise NotImplementedError()

    logdir = get_current_log_path(ns.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    proofs = prover(problems, reasoning_effort=ns.reasoning_effort)
    proofs = [strip_think_simple(proof) for proof in proofs]
    logger.info(f"successfully collected {len(proofs)} proofs from {ns.proof_model}")

    if ns.method == "proofrl":
        evals, verifications = evaluator.verify(prompts, proofs, reasoning_effort=ns.reasoning_effort)
    elif ns.method == "rlvr":
        evals = evaluator(prompts, answers)
        verifications = answers
    elif ns.method == "ttrl":
        evals = evaluator(prompts, answers)
        verifications = [None] * len(evals)

    logger.info("evaluation ended")
    p = sum(evals) / len(evals)
    print(f"Obtained final accuracy: {p}")

    vars_dict = vars(ns)
    vars_dict["accuracy"] = p
    with open(logdir / "logs.json", "w", encoding="utf-8") as f:
        json.dump(vars_dict, f, ensure_ascii=False, indent=2, default=str)

    samples = [
        {
            "problem": problem,
            "proof": proof,
            "eval": eval,
            "verification": verification
        }
        for (problem, proof, eval, verification) in zip(problems, proofs, evals, verifications)
    ]

    with open(logdir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"successfully saved logs to path {logdir}")

def build_parser():
    parser = argparse.ArgumentParser(
        description="ProofRL aims enabling training LRMs on proof problems."
    )

    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=TrlParser)

    p_train = subparsers.add_parser(
        "train",
        dataclass_types=[ScriptArguments, GRPOConfig, ModelConfig],
        add_help=True,
    )

    p_train.add_argument("--method", default="rlvr", choices=["rlvr", "ttrl", "proofrl"], help="the training method switch")
    p_train.add_argument("-em", "--eval_model", help="the model used for evaluation in proofrl (if needed)", default="")
    p_train.add_argument("-d", "--dataset", help="the path to the dataset used for training", default="")
    p_train.add_argument("-ed", "--eval_dataset", help="the path to the dataset used for evaluation", default="")
    p_train.add_argument("--eval_base_url", default="", help="the base url for evaluator")
    p_train.add_argument("--api_key", default="", help="the api key for both prover and evaluator")
    p_train.set_defaults(handler=train, parser=p_train)

    p_eval = subparsers.add_parser(
        "eval",
        add_help=True
    )
    p_eval.add_argument("-ed", "--eval_dataset", help="the path to the dataset used for evaluation", default="")
    p_eval.add_argument("--log_dir", help="the logging directory path", default="eval_logs")
    p_eval.add_argument("-s", "--seed", help="random seed of this project", default=1121)
    p_eval.add_argument("-pm", "--proof_model", help="model that generates proofs for given problems", default="")
    p_eval.add_argument("-em", "--eval_model", help="the model used for evaluation (if needed)", default="")
    p_eval.add_argument("--reasoning_effort", help="the reasoning_effort parameter for some models", default="medium", choices=["minimal", "low", "medium", "high"])
    # p_eval.add_argument("--eval_concurrency", help="the async concurrency in evaluation", default=8)
    p_eval.add_argument("--method", default="rlvr", choices=["rlvr", "ttrl", "proofrl"], help="the training method switch")
    p_eval.add_argument("--prover_base_url", default="", help="the base url for prover")
    p_eval.add_argument("--eval_base_url", default="", help="the base url for evaluator")
    p_eval.add_argument("--api_key", default="", help="the api key for both prover and evaluator")
    p_eval.set_defaults(handler=eval, parser=p_eval)

    return parser

def main():
    parser = build_parser()
    ns, remaining = parser.parse_known_args()

    parsed = ns.parser.parse_args_and_config(return_remaining_strings=True)
    args = list(parsed[:-1]) if isinstance(parsed, tuple) else [parsed]
    ns.handler(*args)

if __name__ == "__main__":
    LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=DATE_FMT
    )
    logger = logging.getLogger(__name__)
    logger.info("Program Started")
    main()
