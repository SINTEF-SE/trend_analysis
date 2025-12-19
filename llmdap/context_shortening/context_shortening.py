import pprint
import torch
import typing

from context_shortening.chunking import chunk_by_headeres_and_clean




class ContextShortener():
    """ General / template context shortener.
    The context shortener reduces the context from the entire paper, to whatever will be fed as context to the llm prompt. This could be a summary, keywords, certain chunks etc."""
    def __init__(self):
        pass
    def set_document(self,document):
        self.document = document
    def set_pydantic_form(self, pydantic_form):
        pass
    def __call__(self, **kwargs):
        raise NotImplementedError


class FullPaperShortener(ContextShortener):
    """ output the whole document """
    def __call__(self, **kwargs):
        return self.document
