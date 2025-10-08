from logger import logger

def query_chain(chain,user_input:str):
    try:
        logger.debug(f"Running chain for input : {user_input}")
        result = chain({"query":user_input})
        response = {
            "response":result["result"],
            "sources":[doc.metadata.get("sources","") for doc in result["source_documents"]]
        }
        logger.debug(f"Chain Response : {response}")
        return response
    except Exception as e:
        logger.exception("Error on Query chain")
        raise

        
