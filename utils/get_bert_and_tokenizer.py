from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
)


AutoList=["bert-base-uncased", "nghuyong/ernie-3.0-base-zh"]

def getBert(logger,bert_name):
    logger.info(f'load {bert_name}')
    if "/".join(bert_name.split("/")[1:]) in AutoList:
        model_config=AutoConfig.from_pretrained(bert_name)
        model_config.output_hidden_states=True
        bert=AutoModel.from_pretrained(bert_name,config=model_config)
    elif "/".join(bert_name.split("/")[1:]) == "facebook/bart-base":
        bert=BartForConditionalGeneration.from_pretrained(bert_name)
        bert.resize_token_embeddings(len(getTokenizer(logger,bert_name)))
    else:
        logger.eror(f'Undefined {bert_name}')
        exit()
    return bert



def getTokenizer(logger,bert_name):
    logger.info(f'load {bert_name} tokenizer')
    if "/".join(bert_name.split("/")[1:]) in AutoList:
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
    elif "/".join(bert_name.split("/")[1:]) == "facebook/bart-base":
        tokenizer = AutoTokenizer.from_pretrained(
            bert_name, use_fast=True, do_lower_case=False
        )
        tokenizer.add_tokens(["<stance>", "<topic>"])
    else:
        logger.error(f"Undefined {bert_name} Tokenizer")
        exit()
    return tokenizer
