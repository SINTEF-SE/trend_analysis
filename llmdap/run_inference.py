import argparse
import yaml
from types import SimpleNamespace





def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """
    with open("llmdap/arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = argument_template[argtype][argname]["default"]
    return parameters

def call_inference(
        schema,
        parsed_paper_text = None,
        raw_xml_paper_text = None,
        paper_path = None,
        paper_url = None,
        graph_traversers=None,
        return_dict_with_context=True,
        traversal_problem_type = None,
        
        **kwargs):
    """
    Fill out the schema for one or several papers.

    Inputs:

    schema: pydantic form that the pipeline should filll out

    include ONE of the following four arguments:
        parsed_paper_text: paper text ready for reading
        raw_xml_paper_text: paper text in xml format (typically starting with "<?xml version="1.0" encoding="UTF-8"?>...)"
        paper_path: local path to paper file in XML format
        paper_url: url to paper file in XML format
    This argument should be either a string, a list of strings, or a dict of strings

    kwargs: Any argument from the arguments.yaml file (e.g. llm, retrievl method and parameters like chunk length)


    output:
    dictionary with the filled form and used contexts for each paper.

    """
    from load_modules import load_modules
    from run_modules import FormFillingIterator

    # prepare arguments
    parameters = add_defaults(kwargs)
    args=SimpleNamespace(**parameters)
    print(args)

    # load stuff
    prepared_kwargs = load_modules(args, inference_schema = schema, graph_traversers = graph_traversers, traversal_problem_type=traversal_problem_type)
    ff_iterator = FormFillingIterator(args, **prepared_kwargs)

    # make the argument into dictionary
    paper_argument = parsed_paper_text or raw_xml_paper_text or paper_path or paper_url
    if type(paper_argument) is str:
        paper_argument = {0: paper_argument}
    elif type(paper_argument) is list:
        paper_argument = {i:p for i,p in enumerate(paper_argument)}
    elif type(paper_argument) is dict:
        pass
    else:
        raise ValueError
    
    outputs = {}
    for key in paper_argument:
        
        # parse/load/fetch argument (to a string paper ready for the llm)
        if parsed_paper_text:
            paper_text = paper_argument[key]
        elif raw_xml_paper_text:
            import dataset_loader
            paper_text = dataset_loader.parse_raw_xml_string(paper_argument[key])
        elif paper_path:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_file_path(paper_argument[key])
        elif paper_url:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_url(paper_argument[key])
        else:
            raise ValueError

        # fill form
        outputs[key] =ff_iterator.fill_single_form(key=key, paper_text=paper_text, pydantic_form=schema, return_dict_with_context=return_dict_with_context)

        # uncomment for quick testing
        if len(outputs)>2: 
            break

    return outputs


class Call_trend_run:
    def __init__(self,):
        self.hf_description_type = "Tags and model card of a Huggingface model"
        self.arx_description_type = "Title and abstract of an arXiv paper"
        self.nl_description_type = "Newsletter item/blurb"

    def load_data(self,n=1):
        from dataset_loader import Arxiv_HF_Newsletters_datasets
        ahd = Arxiv_HF_Newsletters_datasets()
        ahd.prepare()

        hf, arx, newsletters = ahd.get_dict_format(n)

        self.hf = {key: (val, self.hf_description_type) for key, val in hf.items()}
        self.arx = {key: (val, self.arx_description_type) for key, val in arx.items()}

        self.nls = {}
        for nl in newsletters:
            self.nls.update({key: (val, self.nl_description_type) for key, val in nl.items()})

    def load_old_data(self,n=1):
        from dataset_loader import Longterm_Datasets
        ld = Longterm_Datasets()
        ld.prepare()
        arx, nls = ld.get_dict_format(n)
        self.old_arx = {key: (val, self.arx_description_type) for key, val in arx.items()}
        self.old_nls = {}
        for nl in nls:
            self.old_nls.update({key: (val, self.nl_description_type) for key, val in nl.items()})

    def call_run(self,datasets):

        from metadata_schemas.taxonomy_traverser import Traverser, v4_Schema, get_v4_traverser_dict

        traversers = get_v4_traverser_dict() 
        for dataset in datasets:
            output = call_inference(
                    schema = v4_Schema,
                    parsed_paper_text = dataset,
                    graph_traversers = traversers,
                    return_dict_with_context = False,
                    traversal_problem_type = "text2graph",
                    # kwargs
                    save = True,
                    load = True,
                    context_shortener = "full_paper",
                    #ff_model = "4om",
                    ff_model = "41n",
                    #ff_model = "5n",
                    )                                        


if __name__ == "__main__":


    C = Call_trend_run()

    C.load_data(1)#30)
    datasets = [C.hf, C.arx, C.nls]
    C.call_run(datasets)

    #C.load_old_data(30)
    #datasets = [C.old_arx, C.old_nls]
    #C.call_run(datasets)

