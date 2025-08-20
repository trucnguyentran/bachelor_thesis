from sh_recommender import Recommender, ShadowHabtonomics
import uuid
import networkx as nx
import json
import random
import time
import logging

""" A small utility to manually run some SH recommenders"""

def add_logger(filename):
    logger = logging.getLogger('sh_decider')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename)
    formatter = logging.Formatter('%(levelname)s : %(name)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


if __name__ == "__main__":

    #logger = add_logger('decider.log')
    # It turns out this is the order in which the recommendations are returned
    REINFORCED, RELATED, NEW = 0, 1, 2
    REINFORCED_LINEAR, RELATED_LINEAR, NEW_LINEAR = 3, 4, 5
    REINFORCED_MULTIPLY, RELATED_MULTIPLY, NEW_MULTIPLY = 6, 7, 8

    random.seed(time.time())
    #random_user_id = str(uuid.uuid4())
    random_user_id = "e64ec42d-b502-4e8c-89f0-76880c72b220"
    logger = add_logger('./decider.log')
    with open("run_recommender_manually_config.json", "r") as infile:
        """ 
        { 
          "num_repetitions": 5, 
          "initial_block": "preventing-burnout" 
        }
        """
        config = json.load(infile)

    selected_block = config["initial_block"]
    selected_recommender = -1
    selected_element = -1
    choice_log = []
    for i in range(config["num_repetitions"]): 
        message_dict = {Recommender.SBUS_USER_ID_KEY: random_user_id,
                        Recommender.SBUS_BLOCK_CODE_KEY: selected_block}
        this_recommender = Recommender(message_dict)
        topics = this_recommender.sh.conf_topics
        with open("topics.json", "w") as outfile:
            json.dump(topics, outfile, indent=2)
        recommendations = this_recommender.get_knowledge_blocks_rec()
        """ Example recommendations:
        (['managing-sources-of-chronic-stress', 'better-breathing', 'conversation-dealing-with-conflicts'], 
        ['managing-sources-of-chronic-stress', 'conversation-tensing-to-relax', 'preventing-burnout'], 
        ['coping-with-change', 'optimizing-sleep', 'exploring-creativity'])
        """
        
        this_event = {
              "selection": selected_block, 
              "selected_recommender": selected_recommender,
              "selected_element": selected_element,
              "reinforced": recommendations[REINFORCED],
              "related": recommendations[RELATED],
              "new": recommendations[NEW],
              "reinforced_linear": recommendations[REINFORCED_LINEAR],
              "related_linear": recommendations[RELATED_LINEAR],
              "new_linear": recommendations[NEW_LINEAR],
              "reinforced_multiply": recommendations[REINFORCED_MULTIPLY],
              "related_multiply": recommendations[RELATED_MULTIPLY],
              "new_multiply": recommendations[NEW_MULTIPLY],
                }
        logger.debug(this_event)
        choice_log.append(this_event)
        selected_recommender = random.choice(range(9))
        selected_element = random.choice(range(len(recommendations[selected_recommender])))
        selected_block = recommendations[selected_recommender][selected_element]
    with open("manual_sh.log", "w") as outfile:
        json.dump(choice_log, outfile, indent=2)

    #nx.write_gexf(this_recommender.sh.shadow_habtonomics_network, "sh_network.gexf")
