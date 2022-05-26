from typing import Any, Dict, List
from pymongo.collection import Collection

class ModManager:
    def __init__(
        self,
        module: Collection,
    ):
        self.module = module
    
    def save(self) -> None:
        """
        Insert the document into current collection.

        Parameter
        ----------
        document: The document to insert, whose type should be from the schema module.

        """
        raise NotImplementedError
    
    def load(self) -> Dict:
        """
        Load the specified document.

        Parameter
        ----------
        _id: The primary key of the document to load.

        Returns
        ----------
        A dict-like object.
        """
        raise NotImplementedError

    def show(self) -> Any:
        """
        Show the summary or metadata of the specified document. The motivation
        is to provide a quick view to help the user select what they want.
        For example, the evaluation scores of a tree classification model. 

        Parameter
        ----------
        _id: The primary key of the document to show.

        Returns
        ----------
        A materialized view of the specified document.
        """
        raise NotImplementedError

    def query(self) -> List[str]:
        """
        Search documents from current collection by the specified conditions. For example,
        the user can select the model with precision >= .85.

        Parameter
        ----------
        **kwargs: The filter conditions.

        Returns
        ----------
        A list of document _ids that satisfy the filter conditions.
        """
        raise NotImplementedError