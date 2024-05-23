import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_micro, f1_macro
from lm_eval.utils import general_detokenize


_CITATION = """
@inproceedings{chalkidis-etal-2022-lexglue,
    title = "{L}ex{GLUE}: A Benchmark Dataset for Legal Language Understanding in {E}nglish",
    author = "Chalkidis, Ilias  and
      Jana, Abhik  and
      Hartung, Dirk  and
      Bommarito, Michael  and
      Androutsopoulos, Ion  and
      Katz, Daniel  and
      Aletras, Nikolaos",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.297",
    pages = "4310--4330",
}
"""


def build_prompt(user_input):
    # Apply the prompt template and system prompt of LLaMA-2-Chat demo for chat models (NOTE: NO prompt template is required for base models!)
    our_system_prompt = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{user_input} [/INST]"

    return prompt


class CaseHold(Task):
    VERSION = 0
    DATASET_PATH = "lex_glue"
    DATASET_NAME = "case_hold"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        user_input = "Complete the following excerpt from a US court opinion:\n{}: ".format(
            doc["context"]
        )
        return build_prompt(user_input)

    def doc_to_target(self, doc):
        return " {}".format(doc["endings"][doc["label"]])

    def construct_requests(self, doc, ctx):
        lls = [rf.loglikelihood(ctx, " {}".format(choice)) for choice in doc["endings"]]
        return lls

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold, "f1_micro": (gold, pred), "f1_macro": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "f1_micro": True, "f1_macro": True}

    def aggregation(self):
        return {"acc": mean, "f1_micro": f1_micro, "f1_macro": f1_macro}


class SCOTUS(Task):
    VERSION = 0
    DATASET_PATH = "AdaptLLM/law-tasks"
    DATASET_NAME = "SCOTUS"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["input"]

    def doc_to_target(self, doc):
        return " {}".format(doc["options"][doc["gold_index"]])

    def construct_requests(self, doc, ctx):
        lls = [rf.loglikelihood(ctx, " {}".format(choice)) for choice in doc["options"]]
        return lls

    def process_results(self, doc, results):
        results = [result[0] for result in results]
        gold = doc["gold_index"]
        pred = np.argmax(results)
        return {"acc": pred == gold, "f1_micro": (gold, pred), "f1_macro": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "f1_micro": True, "f1_macro": True}

    def aggregation(self):
        return {"acc": mean, "f1_micro": f1_micro, "f1_macro": f1_macro}


class UNFAIR_ToS(Task):
    VERSION = 0
    DATASET_PATH = "AdaptLLM/law-tasks"
    DATASET_NAME = "UNFAIR_ToS"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["input"]

    def doc_to_target(self, doc):
        return " {}".format(doc["options"][doc["gold_index"][0]])

    def construct_requests(self, doc, ctx):
        lls = [rf.loglikelihood(ctx, " {}".format(choice)) for choice in doc["options"]]
        return lls

    def process_results(self, doc, results):
        results = [result[0] for result in results]
        gold_indexes = doc["gold_index"]
        pred = np.argmax(results)

        gold = gold_indexes[0]
        if pred in gold_indexes:
            gold = pred

        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
