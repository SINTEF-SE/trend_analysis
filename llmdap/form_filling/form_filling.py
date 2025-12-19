import torch
from pydantic import constr
from pydantic_core._pydantic_core import ValidationError as pydantic_ValidationError
import pydantic
import typing
import json
import copy
import pprint
import numpy as np
from difflib import SequenceMatcher
import openai


from form_filling import regex_handling



def make_FormFillPrompt(context, 
                        answer_field_name=None, 
                        answer_field_type=None, 
                        answer_field_description=None, 
                        prompt_for_reasoning=False,
                        reason=None
                        ):
    # get prompt for sequential form filling (i.e. one field at a time), field-agnistic.
    prompt = f"""
You are to fill out values of a form based on some context from a part of a scientific paper or document.
Use only the context to reply. If the answer is not in the context directly, make a qualified guess based on what is in the context.
"""
    prompt += "\n"

    # Add context
    prompt += f"context: {{{{{{{context}}}}}}}\n\n"

    # Add field info
    if not answer_field_name is None:
        prompt += f"answer field name: {{{{{{{answer_field_name}}}}}}}\n\n"
    if not answer_field_type is None:
        prompt += f"answer field type: {{{{{{{answer_field_type}}}}}}}\n\n"
    if not answer_field_description is None:
        prompt += f"answer field description: {{{{{{{answer_field_description}}}}}}}\n\n"

    # Add reasoning step
    if prompt_for_reasoning:
        prompt += "Reasoning step: {{{Let's think step by step. "
        return prompt
    if not reason is None:
        prompt += f"Reasoning step: {{{{{{{reason}}}}}}}\n\n"

    prompt += "Answer: {{{"
    return prompt

def make_graph2graph_traversal_prompt(term,
                          current_path,
                          child_nodes,
                          parent_nodes=None,
                          sibling_nodes=None,
                          term_path=None,
                          **kwargs
                          ):
    prompt = """Your task is to add a term to a taxonomy or ontology. You will do this in several steps.
Your first task is to find the correct placement of the new term. This is done by iteratively traversing the ontology.
At each iteration, you are given the name of the current node, as well as its """
    if parent_nodes is None:
        prompt+="child nodes."
        assert sibling_nodes is None
    elif sibling_nodes is None:
        prompt+="parent and child nodes."
    else:
        prompt+="parent, sibling and child nodes."
    prompt +="""
You are also given the absolute position of the current node (i.e. the name of all its anscestors), for context.
You must choose which of these are most relevant for the new term.
- If you choose the current node, you will then move on to the next step, which is to decide if the new term represents the same concept as this node and should be merged with it, or if it should be added as a child of this node.
- If you choose any of the other nodes """
    if parent_nodes is None:
        prompt+="(child nodes)"
    elif sibling_nodes is None:
        prompt+="(parent or child nodes)"
    else:
        prompt+="(parent, sibling or child nodes)"
    prompt +=""", you will move position to that node, and redo this task from there. This way you can iteratively traverse through the ontology. You should aim to end up at the most relevant node, and as specific as possible while still being the appropriate parent node or merged node to the new term.

The concrete variables for the task are listed here:
"""
    prompt += f"New term: {term}\n"
    if not term_path is None:
        prompt += f"Position of the term in its old taxonomy, provided here for context): {term_path}\n"
    prompt += f"Current node: {current_path[-1]}\n"
    prompt += f"Absolute position of current node: {'/'.join(current_path)}\n"

    if not sibling_nodes is None:
        prompt += f"Parent node(s): {parent_nodes}\n"
    if not sibling_nodes is None:
        prompt += f"Sibling nodes: {sibling_nodes}\n"
    prompt += f"Child nodes: {child_nodes}\n"
    
    prompt += "\nPlease provide the most correct position of the new term below.\n"
    prompt += "Answer: "

    #print("::::")
    #print(prompt)
    #print("::::")

    return prompt


def make_text2graph_traversal_prompt(text,
                          current_path,
                          child_nodes,
                          parent_nodes=None,
                          sibling_nodes=None,
                          text_type=None,
                          **kwargs
                          ):
    prompt = """Your task is to label a textual document with a tag from a taxonomy or ontology. This is done by iteratively traversing the ontology.
At each iteration, you are given the name of the current node, as well as its """
    if parent_nodes is None:
        prompt+="child nodes."
        assert sibling_nodes is None
    elif sibling_nodes is None:
        prompt+="parent and child nodes."
    else:
        prompt+="parent, sibling and child nodes."
    prompt +="""
You must choose which of these are most relevant for the document.
- If you choose the current node, the document is labeled with this term.
- If you choose any of the other nodes """
    if parent_nodes is None:
        prompt+="(child nodes)"
    elif sibling_nodes is None:
        prompt+="(parent or child nodes)"
    else:
        prompt+="(parent, sibling or child nodes)"
    prompt +=""", you will move position to that node, and redo this task from there. This way you can iteratively traverse through the ontology. You should aim to end up at the most relevant node, and as specific as possible while still being correct (just choose the current node if none of the child nodes are appropriate)."""
    

    if "Not relevant" in child_nodes:
        prompt += f"\nNote: If the ontology ({current_path[0]}) is not relevant to the document, or if the document does not contain information relevant to the ontology, please use the 'Not relevant' option. In other words, use 'Not relevant' to avoid guessing labels not inferrable from the document." # Saying this 3 times to hopefully make it happen
    if "Other" in child_nodes:
        prompt += f"\nNote: If none of the child nodes are correct labels for the documents, but you still think the current node ({current_path[-1]}) is correct, please answer '{current_path[-1]}'. If you think the current node is not actually correct, please use the 'other' option."
    prompt +="""

The concrete variables for the task are listed here:
"""
    if not text_type is None:
        prompt += f"Type of document: {text_type}\n"
    prompt += f"Document: \n{text}\n(document finished)\n"
    prompt += f"Current node: {current_path[-1]}\n"
    prompt += f"Absolute position of current node: {'/'.join(current_path)}\n"
    if not sibling_nodes is None:
        prompt += f"Parent node(s): {parent_nodes}\n"
    if not sibling_nodes is None:
        prompt += f"Sibling nodes: {sibling_nodes}\n"
    prompt += f"Child nodes: {child_nodes}\n"
    
    prompt += "\nPlease provide the most relevant label below:\n"
    prompt += "Answer: "

    #print("::::")
    #print(prompt)
    #print("::::")

    return prompt

def make_merge_or_subnode_prompt(
                          term,
                          current_path,
                          child_nodes=None,
                          sibling_nodes=None,
                          parent_nodes=None,
                          term_path=None,
                          **kwargs
                          ):
    prompt = """Your task is to add a term to a taxonomy or ontology. You will do this in several steps.
You have already chosen which node in the ontology to connect the term to. You must now decide if the new term should be a child node of the chosen node, or if it should be merged with the chosen node. If the new term represents the same concept as the chosen node, write "merge" in the field below. If it represents a more specific concept, write "child".
These are the only two options, choose what is most appropriate. 
For context; the goal is not the creation/extension of the ontology, but to standardize metadata for files, that are currently marked with tags that are not connected to the ontology. The decision will determine which tag from the (possibly expanded) ontology is used to describe files labeled with the new term.
Thus, merging tags that are not exactly the same concept will not ruin the integrity of the ontology.

The concrete variables for the task are listed here:
"""
    prompt += f"New term: {term}\n"
    if not term_path is None:
        prompt += f"Position of the term in its old taxonomy, provided here for context): {term_path}\n"
    prompt += f"Current node: {current_path[-1]}\n"
    prompt += f"Position of chosen node in the ontology: {'/'.join(current_path)}\n"
    if child_nodes:
        prompt += "\nTo give context of how the ontology is structured, we also provide the child, sibling and parent nodes of the chosen node:\n"
        prompt += f"Child nodes: {child_nodes}\n"
        prompt += f"Sibling nodes: {child_nodes}\n"
        prompt += f"Parent node(s): {parent_nodes}\n"

    prompt += """\nPlease provide the most correct choice of what to do with the term relative to the chosen node, either "child" or "merge", below.\n"""
    prompt += "Answer: "


def get_constraints_from_field(field):
    """ get field_type, and any min/max length if its a constr """
    field_type = field.annotation
    metadata = field.metadata

    # determine any constraints
    if len(metadata):
        constraints = metadata[0]
        assert field_type == str # this is the only constrained field implemented for now
        assert type(constraints) == pydantic.types.StringConstraints
        min_l, max_l = constraints.min_length, constraints.max_length
    else:
        min_l, max_l = None, None
    return field_type, min_l, max_l



class FieldFiller:
    """
    Module for filling out a single field in a pydantic form.
    One FieldFiller is made per schema field, but used for multiple documents.
    document shortener is fed in the forwards fuction.
    """
    def __init__(self, answer_generator, answer_in_quotes=False, verbose=False):

        self.answer_generator = answer_generator
        self.answer_in_quotes = answer_in_quotes
        self.verbose = verbose

    def forward(self, prompt_input, field_type, prompt_function = make_FormFillPrompt):

        # generate answer
        prompt = prompt_function(**prompt_input)
        if self.verbose:
            print("Prompt::::")
            print(prompt)
            print("::::")
        answer = self.answer_generator(prompt)
        if self.verbose:
            print("Generated answer:", answer)


        #print("generated answer", answer, type(answer))
        assert type(answer) is str, (answer, type(answer))


        if self.answer_in_quotes:
            answer=answer[1:-1] # remove quotes. 
        return parse_single_output(field_type, answer)

def parse_single_output(field_type, stringoutput, answer_in_quotes = None):
    # parse string output into the given type
    try:
        if getattr(field_type, "__origin__", None) is typing.Literal:
            output = str(stringoutput)
        else:
            output = field_type(stringoutput)
    except ValueError as e:
        print("\n\n\n")
        print("!!!")
        print("WARNING: FAILED TO READ STRING:", stringoutput, "inserting empty value instead")
        print("Check regex rules - they seem to allow unparsable output for class:", field_type)
        print("answer in quotes?", answer_in_quotes)
        #print(e)
        raise e
        print("\n\n\n")
        output = field_type()
    return output

class SequentialFormFiller:
    """
    Class for iterating through a pydantic schema, and predict each field sequentially.
    Uses outlines to ensure correct field types.
    Dspy is wrapped around outlines, to enable optimization.
    """
    def __init__(self,
                 outlines_llm,
                 outlines_sampler,
                 pydantic_form = None,
                 answer_in_quotes = True,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.llm_model = outlines_llm
        self.sampler = outlines_sampler
        self.verbose = verbose
        self.max_tokens = max_tokens

        self.answer_in_quotes=answer_in_quotes
        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        """ Prepares generator for each field typ in the pydantic form """
        self.pydantic_form = pydantic_form
        self.fields = pydantic_form.__fields__ 

        self.outlines_generators = {}

        if self.verbose:
            print("Generating regex generators...")

        # iterate through fields
        for fieldname in self.fields:
            field_type, min_l, max_l = get_constraints_from_field(self.fields[fieldname])

            # only make a new generator if it is not equal to one already generated
            if not (field_type, min_l, max_l) in self.outlines_generators:
                outlines_generator = regex_handling.make_constrained_generator(
                        llm_model=self.llm_model,
                        field_type=field_type,
                        min_l=min_l,
                        max_l=max_l,
                        answer_in_quotes=self.answer_in_quotes,
                        sampler = self.sampler)
                self.outlines_generators[(field_type, min_l, max_l)] = outlines_generator

        if self.verbose:
            print("Finished generating regex generators.")

        self.prepare_field_fillers()

    def prepare_field_fillers(self):
        self.field_fillers = {}
        for fieldname in self.fields:
            field = self.fields[fieldname]
            field_type, min_l, max_l = get_constraints_from_field(field)
            generator = self.outlines_generators[(field_type, min_l, max_l)]
            self.field_fillers[fieldname] = FieldFiller(
                    answer_generator = generator,
                    verbose = self.verbose,
                    answer_in_quotes = self.answer_in_quotes,
                    )

    def re_set_pydantic_form(self,pydantic_form):
        """after shufling literal values, there is no need to remake field filleds since they do not use order"""
        if self.pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
        self.pydantic_form = pydantic_form
        self.fields = pydantic_form.__fields__ 


    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields)

        output_dict = {}
        self.contexts = {}

        # iterate through fields
        if self.verbose:
            print("--INFO--:starting to iterate through fields")
        for fieldname in self.fields:
            field = self.fields[fieldname]
            field_type = field.annotation


            # make prompt input
            prompt_input = {
                           "context":None,
                           "answer_field_name":fieldname,
                           "answer_field_description":field.description,
                           "answer_field_type":str(field_type),
                            }
            context = get_context(**prompt_input)
            prompt_input["context"] = context
            self.contexts[fieldname] = context

            # generate output
            output = self.field_fillers[fieldname].forward(prompt_input, field_type)

            output_dict[fieldname] = output

        if self.verbose:
            print("--INFO--: fields iterated")

        try:
            output = pydantic_form(**{name : val.__str__() for name, val in output_dict.items()})
        except pydantic_ValidationError: # outlines seem to allow non-allowable strings in rare occasions. Workaround: just choose closest allowable answer

            output_dict = {name : val.__str__() for name, val in output_dict.items()}
            for name, predicted_string in output_dict.items():

                field = self.fields[name]
                field_type = field.annotation
                if typing.get_origin(field_type) == typing.Literal: # i have only seen this problem in Literal fields
                    allowed_answers = field_type.__args__
                    if not predicted_string in allowed_answers: # only alter the field(s) with the problem
                        
                        best_similarity  = -1.0
                        best_ans = None
                        for ans in allowed_answers:
                            similarity = SequenceMatcher(None, ans, predicted_string).ratio()
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_ans = ans

                        output_dict[name] = best_ans
                        
                        print("!!!!! Failed to generate allowable answer. Finding closest allowable answer instead")
                        print("Predicted answer:", predicted_string, "while closest match is", best_ans)
                        # log this 
                        with open("output_log.txt", "a") as file:
                            file.write(f"\nfound closest match: {predicted_string} -> {best_ans}\n")
            output = pydantic_form(**output_dict)

        torch.cuda.empty_cache()
        return output

#    def deepcopy(self):
#        """ avoid copying llm_model """
#        if self.verbose:
#            print("--INFO--: deepcopy called")
#        lm = self.llm_model
#        self.llm_model = None
#        copy = super().deepcopy()
#        self.llm_model = lm
#        copy.llm_model = lm
#        return copy
#
#    def reset_copy(self):
#        """ avoid copying llm_model """
#        if self.verbose:
#            print("--INFO--: reset_copy called :)")
#        lm = self.llm_model
#        self.llm_model = None
#        copy = super().reset_copy()
#        self.llm_model = lm
#        copy.llm_model = lm
#        return copy


def get_subschema(original_schema: pydantic.BaseModel, exclude_fields: list = [], remove_maxlength= False):
    """Get a pydantic form with fewer fields"""

    # Extract the fields from the original schema
    original_fields = original_schema.__annotations__
    # Filter the fields based on the provided list
    new_fields = {}
    for field in original_fields:
        if not field in exclude_fields:

            properties = original_schema.schema()["properties"][field]
            #print(properties)
            if remove_maxlength:
                # this is needed for openai api
                if "maxLength" in properties:
                    properties.pop("maxLength") # maxlength is simply ignored

                new_fields[field] = (original_schema.__fields__[field].annotation, pydantic.Field(**properties))
            else:
                new_fields[field] = (original_fields[field], pydantic.Field(**properties))

    # Create a new model with the filtered fields
    NewSchema = pydantic.create_model('NewSchema', **new_fields)
    return NewSchema

###
# openai
### 


class OpenAIFormFiller:
    """
    """
    def __init__(self,
                 model_id,
                 pydantic_form=None,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.model_id = model_id
        self.verbose = verbose


        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        self.pydantic_form = pydantic_form
        self.fields = pydantic_form.__fields__ 
    def re_set_pydantic_form(self, pydantic_form):
        self.set_pydantic_form(pydantic_form)

    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields, remove_maxlength= True)


        output_dict = {}


        # prepare context
        context = get_context()
        self.contexts = context
        if self.verbose:
            print("context:")
            print(context)
            #print("schema:")
            #print(pydantic_form.schema())


        # generate answer
        completion = openai.beta.chat.completions.parse(
                                                        model = self.model_id,
                                                        messages = [
                                                            {
                                                                "role":"user",
                                                                "content": make_FormFillPrompt(context = context),
                                                            }
                                                        ],
                                                        response_format = pydantic_form,
                                                        )
        if len(completion.choices) > 1:
            print(completion)
            print(completion.model_dump)
            print(completion.choices)
            quit()
        answer = completion.choices[0].message.content
        if verbose:
            print("------------called openai, model=", completion.model)


        
        if self.verbose:
            print("!")
            print("!")
            print("!")
            print("answer generated!")
            print(type(answer))
            print(answer)

        output_dict = json.loads(answer)
        output = pydantic_form(**output_dict)

        return output 



###
# openai sequential
###

def openAIFieldFiller(prompt_input, # used for retrieval and generation
                      model_id,
                      subschema, # used for generation, and NOT retrieval
                      prompt_function=make_FormFillPrompt,
                      verbose=False
                      ):



        # generate answer
        generation_kwargs = dict(model = model_id,
                                 messages = [{"role":"user", "content": prompt_function(**prompt_input)}],
                                 response_format = subschema)
        try:
            completion = openai.beta.chat.completions.parse(**generation_kwargs)
        except openai.RateLimitError:
            import time
            print("\n ... sleeping a bit to avoid OpenAI rate limit) ...")
            time.sleep(15)
            completion = openai.beta.chat.completions.parse(**generation_kwargs)

        if len(completion.choices) > 1:
            print(completion)
            print(completion.model_dump)
            print(completion.choices)
            quit()
        answer = completion.choices[0].message.content
        if verbose:
            print("------------called openai, model=", completion.model)

        output_dict = json.loads(answer)
        if len(output_dict) != 1:
            print("!!!!!", output_dict)
            raise ValueError
        for key in output_dict:
            value = output_dict[key]
        return value



class OpenAISequentialFormFiller:
    """
    """
    def __init__(self,
                 model_id,
                 pydantic_form = None,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.model_id = model_id
        self.verbose = verbose
        self.max_tokens = max_tokens
        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        self.pydantic_form = pydantic_form
        self.fields = pydantic_form.__fields__ 
    def re_set_pydantic_form(self, pydantic_form):
        self.set_pydantic_form(pydantic_form)

    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields)

        fields = pydantic_form.__fields__
        output_dict = {}
        self.contexts = {}

        # iterate through fields
        if self.verbose:
            print("--INFO--:starting to iterate through fields")
        for fieldname in fields:
            field = fields[fieldname]
            field_type = field.annotation


            # make prompt input
            prompt_input = {
                           "context":None, 
                           "answer_field_name":fieldname, 
                           "answer_field_description":field.description, 
                           "answer_field_type":str(field_type),
                            }

            # prepare context
            context = get_context(**prompt_input)
            self.contexts[fieldname] = context
            prompt_input["context"] = context
        
            all_other_fields = list(self.pydantic_form.__fields__.keys())
            all_other_fields.remove(fieldname)
            subschema = get_subschema(self.pydantic_form, exclude_fields = all_other_fields, remove_maxlength= True) # for generation, remove the stuff openai cant handle

            # generate output
            output = openAIFieldFiller(
                      prompt_input = prompt_input,
                      model_id = self.model_id,
                      subschema = subschema,
                      verbose=self.verbose,
                      )

            output_dict[fieldname] = output

        if self.verbose:
            print("--INFO--: fields iterated")

        # make pytantic object and return
        output = pydantic_form(**{name : val.__str__() for name, val in output_dict.items()})
        torch.cuda.empty_cache()
        return output






class AdaptiveFormFiller:
    """
    Class for traversing through a graph, and predict each step based on the previous.
    """
    def __init__(self,
                 openai_model_id = None,
                 outlines_llm = None,
                 outlines_sampler = None,
                 pydantic_form = None,
                 graph_traversers = None,
                 traversal_type = None,
                 traversal_max_steps = None,
                 answer_in_quotes = True,
                 max_tokens = 50,
                 verbose = False,
                 problem_type = "text2graph"
                 ):
        self.openai_model_id = openai_model_id
        self.llm_model = outlines_llm
        self.sampler = outlines_sampler
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.starting_traversal_type = traversal_type
        self.traversal_max_steps = traversal_max_steps

        self.answer_in_quotes=answer_in_quotes
        self.pydantic_form = pydantic_form
        self.graph_traversers = graph_traversers
        self.fields = pydantic_form.__fields__ 
        self.field_fillers = {}
        self.problem_type = problem_type


    def prepare_field_filler(self, field, current_path_string):
        field_type = field.annotation

        # only make a new generator if it is not equal to one already generated
        if not current_path_string in self.field_fillers:
            if self.verbose:
                print("Generating regex generator...")

            outlines_generator = regex_handling.make_constrained_generator(
                    llm_model=self.llm_model,
                    field_type=field_type,
                    min_l=None,
                    max_l=None,
                    answer_in_quotes=self.answer_in_quotes,
                    sampler = self.sampler)

            self.field_fillers[current_path_string] = FieldFiller(
                    answer_generator = outlines_generator,
                    verbose = self.verbose,
                    answer_in_quotes = self.answer_in_quotes,
                    )
            if self.verbose:
                print("Finished generating regex generator:", current_path_string)



    def recursive_forward(self, get_context, exclude_fields = []):
        current_path = self.current_traverser.current_path
        current_path_string = "__".join(current_path)



        # make prompt input
        prompt_input = { "current_path":current_path,
                       "allowed_answers":[*self.current_traverser.get_child_nodes(), current_path[-1]],
                       "child_nodes":self.current_traverser.get_child_nodes(),
                        }
        if self.traversal_type in ["free", "vertical"]:
            prompt_input.update({
                "parent_nodes":self.current_traverser.get_parent_nodes()
                })
        if self.traversal_type == "free":
            prompt_input.update({
                "sibling_nodes":self.current_traverser.get_sibling_nodes(),
                })


        if self.problem_type == "text2graph":
            text, text_type = get_context()
            prompt_input.update({
                           "text":text,
                           "text_type":text_type,
                })
        elif self.problem_type == "graph2graph":
            term_path = get_context()
            term = term_path.split("/")[-1]
            prompt_input.update({
                           "term":term,
                           "term_path":term_path,
                })
        else:
            raise ValueError

        # generate output
        try:
            if self.openai_model_id is None: # use outlines model
                current_field = self.current_traverser.get_field()
                self.prepare_field_filler(current_field, current_path_string)
                field_type = current_field.annotation
                output = self.field_fillers[current_path_string].forward(
                        prompt_input, 
                        field_type, 
                        prompt_function=make_graph2graph_traversal_prompt if self.problem_type == "graph2graph" else make_text2graph_traversal_prompt,
                        )

            else: # use openai
                output = openAIFieldFiller(
                        prompt_input = prompt_input,
                        model_id = self.openai_model_id,
                        subschema = self.current_traverser.get_pydantic_form(),
                        verbose=self.verbose,
                        prompt_function=make_graph2graph_traversal_prompt if self.problem_type == "graph2graph" else make_text2graph_traversal_prompt,
                        )
        except StopIteration:
            return


        assert type(output) is str 
        print(output)

        if output == self.current_traverser.current_path[-1]:
            return

        direction = self.current_traverser.move(output)
        self.traversal_steps.append(direction+" "+output)

        if output == "Other":
            return

        if len(self.traversal_steps) >= self.traversal_max_steps and self.traversal_type != "down":
            print("----Max travesal steps reached. Reverting to downward traversal.")
            self.traversal_type = "down"
            self.current_traverser.set_traversal_type("down")
        self.recursive_forward(get_context)






    def single_traverser_forward(self, get_context, exclude_fields = []):
        self.current_traverser.reset_position() # can add another node to start from (from e.g. similarity match)
        self.current_traverser.set_traversal_type(self.starting_traversal_type)
        self.traversal_steps = []
        self.traversal_type = self.starting_traversal_type

        # traverse recursively
        self.recursive_forward(get_context)
        path = self.current_traverser.current_path
        #path = "/".join(path)
        print("---- finished traversal. Steps made:", self.traversal_steps)

        # merge or child node
        if self.problem_type == "graph2graph":
            raise NotImplementedError
        return path




    def forward(self, get_context, exclude_fields = []):
        if type(self.graph_traversers) == dict:
            output_dict = {}
            for key in self.graph_traversers:
                self.current_traverser = self.graph_traversers[key]
                output_dict[key] = self.single_traverser_forward(get_context, exclude_fields)




        else:
            self.current_traverser = self.graph_traversers
            path = self.single_traverser_forward(get_context, exclude_fields)
            output_dict = {next(iter(self.fields.keys())) : path}

        print(f"finished traversing::: ")
        print(get_context()[0])
        print("generated dict:")
        print(output_dict)
        filled_form = self.pydantic_form(**output_dict)
        torch.cuda.empty_cache()
        return filled_form
