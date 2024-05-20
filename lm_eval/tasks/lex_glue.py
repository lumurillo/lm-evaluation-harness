import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
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


class CaseHold(Task):
    VERSION = 0
    DATASET_PATH = "lex_glue"
    DATASET_NAME = "case_hold"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

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
        return "Complete the following excerpt from a US court opinion:\n{}:".format(
            doc["context"]
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        print(f"ctx: {ctx}")
        ll_0, _ = rf.loglikelihood(ctx, " 0")
        ll_1, _ = rf.loglikelihood(ctx, " 1")
        ll_2, _ = rf.loglikelihood(ctx, " 2")
        ll_3, _ = rf.loglikelihood(ctx, " 3")
        ll_4, _ = rf.loglikelihood(ctx, " 4")
        return ll_0, ll_1, ll_2, ll_3, ll_4

    def process_results(self, doc, results):
        print(f"Processing the results: {results}")
        gold = doc["label"]
        pred = np.argmax(results)
        print(f"Prediction: {pred}")
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
