"""ViLP task implementation."""

from lmms_eval.api.task import ConfigurableTask


class ViLPTask(ConfigurableTask):
    """Custom task class for ViLP benchmark.

    ViLP has 3 images per question, and we need to evaluate each
    image separately to compute ViLP-F/ViLP-P scores and priors.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize ViLP task."""
        super().__init__(**kwargs)
        self._expanded_docs_cache: dict[str, list] = {}

    def _expand_doc(self, doc: dict) -> list[dict]:
        """Expand a single doc with 3 images into 3 separate docs.

        Args:
            doc: Original document with image1-3 and answer1-3

        Returns:
            List of 3 documents, one for each image
        """
        expanded_docs = []

        for i in range(1, 4):
            image_key = f"image{i}"
            answer_key = f"answer{i}"

            # Skip if image or answer is missing
            if image_key not in doc or answer_key not in doc or doc[image_key] is None or doc[answer_key] is None:
                continue

            # Create a new doc for this image
            new_doc = {
                "question": doc["question"],
                image_key: doc[image_key],
                answer_key: doc[answer_key],
                "_image_idx": i,
                "_original_idx": doc.get("_original_idx", 0),
            }
            expanded_docs.append(new_doc)

        return expanded_docs

    def _process_docs(self, docs: list[dict]) -> list[dict]:
        """Process and expand documents.

        Args:
            docs: Original documents from dataset

        Returns:
            Expanded documents with one entry per image
        """
        expanded_docs = []

        for idx, doc in enumerate(docs):
            # Add original index for tracking
            doc["_original_idx"] = idx

            # Expand into 3 docs (one per image)
            expanded = self._expand_doc(doc)
            expanded_docs.extend(expanded)

        return expanded_docs

    def validation_docs(self) -> list[dict]:
        """Return validation documents (expanded)."""
        if self.has_validation_docs():
            if "validation" not in self._expanded_docs_cache:
                docs = list(self.dataset["validation"])
                self._expanded_docs_cache["validation"] = self._process_docs(docs)
            return self._expanded_docs_cache["validation"]
        return []

    def test_docs(self) -> list[dict]:
        """Return test documents (expanded)."""
        if self.has_test_docs():
            if "test" not in self._expanded_docs_cache:
                docs = list(self.dataset["test"])
                self._expanded_docs_cache["test"] = self._process_docs(docs)
            return self._expanded_docs_cache["test"]
        return []

    def train_docs(self) -> list[dict]:
        """Return train documents (expanded)."""
        if self.has_training_docs():
            if "train" not in self._expanded_docs_cache:
                docs = list(self.dataset["train"])
                self._expanded_docs_cache["train"] = self._process_docs(docs)
            return self._expanded_docs_cache["train"]
        return []
