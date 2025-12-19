# schema with single field, free string but max 100 chars.

from pydantic import BaseModel, Field, constr

class Metadata_form(BaseModel):

    fieldname : constr(max_length=100) = Field(description = "") 

