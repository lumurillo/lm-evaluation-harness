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


class CaseHold(Task):
    VERSION = 0
    DATASET_PATH = "lex_glue"
    DATASET_NAME = "case_hold"
    NUM_CHOICES = 5

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
        return " {}".format(doc["endings"][doc["label"]])

    def construct_requests(self, doc, ctx):
        endings = doc["endings"]
        lls = [rf.loglikelihood(ctx, " {}".format(endings[choice])) for choice in range(self.NUM_CHOICES)]
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
    DATASET_PATH = "lex_glue"
    DATASET_NAME = "scotus"

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
        return "Given the following opinion from the Supreme Court of USA (SCOTUS):\n{}\nThe relevant issue area is:".format(
            doc["text"]
        )

    def doc_to_target(self, doc):
        return " {}".format({
            0: "Criminal Procedure", 
            1: "Civil Rights", 
            2: "First Amendment", 
            3: "Due Process", 
            4: "Privacy",
            5: "Attorneys",
            6: "Unions",
            7: "Economic Activity",
            8: "Judicial Power",
            9: "Federalism",
            10: "Interstate Relations",
            11: "Federal Taxation",
            12: "Miscellaneous",
            13: "Private Action"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_criminal_procedure, _ = rf.loglikelihood(ctx, " Criminal Procedure")
        ll_civil_rights, _ = rf.loglikelihood(ctx, " Civil Rights")
        ll_first_amendment, _ = rf.loglikelihood(ctx, " First Amendment")
        ll_due_process, _ = rf.loglikelihood(ctx, " Due Process")
        ll_privacy, _ = rf.loglikelihood(ctx, " Privacy")
        ll_attorneys, _ = rf.loglikelihood(ctx, " Attorneys")
        ll_unions, _ = rf.loglikelihood(ctx, " Unions")
        ll_economic_activity, _ = rf.loglikelihood(ctx, " Economic Activity")
        ll_judicial_power, _ = rf.loglikelihood(ctx, " Judicial Power")
        ll_federalism, _ = rf.loglikelihood(ctx, " Federalism")
        ll_interstate_relations, _ = rf.loglikelihood(ctx, " Interstate Relations")
        ll_federal_taxation, _ = rf.loglikelihood(ctx, " Federal Taxation")
        ll_miscellaneous, _ = rf.loglikelihood(ctx, " Miscellaneous")
        ll_private_action, _ = rf.loglikelihood(ctx, " Private Action")
        return (
            ll_criminal_procedure, ll_civil_rights, ll_first_amendment, ll_due_process, 
            ll_privacy, ll_attorneys, ll_unions, ll_economic_activity, ll_judicial_power, 
            ll_federalism, ll_interstate_relations, ll_federal_taxation, ll_miscellaneous, 
            ll_private_action
        )  

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold, "f1_micro": (gold, pred), "f1_macro": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "f1_micro": True, "f1_macro": True}

    def aggregation(self):
        return {"acc": mean, "f1_micro": f1_micro, "f1_macro": f1_macro}
