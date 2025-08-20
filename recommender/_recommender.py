#! /usr/bin/env python
"""
    This file has two classes
    1. createSHInput: This class will digest the servicebus message
            and store the useful information from it. It will then look up
            the past history of blocks completed by the user. Finally it
            creates a list of topics completed by the user and outputs it as
            dataframe which is the required format for the SH decider class.
    2. getRecommendations: This class will call the createSHInput class to get
            the required input for the decider class. It then gets the decider
            input to output the SH recommendations per decider type. Finally it
            formats the output above into a json structure which the backend
            team has requested for.
"""
import logging

import json
import os
import pickle
import uuid
from datetime import datetime
import random
import pandas as pd
import pkg_resources

from sh_recommender._decider import Decider
from sh_recommender import ShadowHabtonomics

logger = logging.getLogger(__package__)


class SHInput:
    """This class will digest the servicebus message and also take into account
    the blocks already finished by the user and thus eventually prepare the
    input which will be fed into the SH Recommender.
    Args:
        completed_blocks_filename: The dataframewhich stores the blocks
                                    complete by the user historically
        keys_filename: The various keys and secrets needed are stored here
                        particularly the conf topic names
        service_bus_message: The servicebusmessage from the sh_recommender
                            subscription
    Returns:
        The input in the format required by the Decider class
    """

    def __init__(self,
                 base_path: str,
                 completed_blocks_filename: str,
                 conf_cms_filename: str,
                 keys_filename: str,
                 service_bus_message: dict,
                 ):
        self._service_bus_message = service_bus_message
        self._user_id = ""
        self._block_completed = ""
        self._previous_blocks_completed, self._updated_blocks = [], []
        self._completion_state = ""
        self._completed_topics = []
        self.base_path = base_path
        self.completed_blocks_filename = completed_blocks_filename
        self.conf_cms_filename = conf_cms_filename
        self.json_keys_file = os.path.join(base_path, keys_filename)
        self.conf_cms_file = pd.read_csv(os.path.join(base_path,
                                                      conf_cms_filename))
        self.completed_blocks_filepath = os.path.join(base_path,
                                                      completed_blocks_filename)

        self.df_completed: pd.DataFrame = self.get_completed_blocks()

    def get_completed_blocks(self):
        """Reads the csv file with the history of blocks finished if exists.
        If it doesn;t exist, return an empty dataframe with the expected syntax."""
        if os.path.exists(self.completed_blocks_filepath):
            return pd.read_csv(self.completed_blocks_filepath)
        else:
            return pd.DataFrame(
                {"UserId": pd.Series(dtype="object"),
                 "completed_blocks": pd.Series(dtype="object")})

    def get_user_data(self):
        self._user_id = self._service_bus_message[Recommender.SBUS_USER_ID_KEY]
        self._block_completed = str(self._service_bus_message[Recommender.SBUS_BLOCK_CODE_KEY]).strip()
        self._completion_state = 'true'

    def get_previous_block_completion(self):
        """This function will check the historically completed blocks by the
        user and if the user is absent from the data it adds an entry for the
        user."""
        self.get_user_data()

        # To accomodate new users create a new entry for the user
        if self._user_id not in self.df_completed["UserId"].values:
            self.df_completed = self.df_completed.append(
                pd.DataFrame([[self._user_id, self._block_completed]], columns=["UserId", "completed_blocks"]))
            self._previous_blocks_completed = [self._block_completed]
        else:
            # get last completed block for this user
            self._previous_blocks_completed = self.df_completed.loc[self.df_completed["UserId"] == self._user_id,
                                                                    "completed_blocks"].values[0]
            self._previous_blocks_completed = [block.strip() for block in
                                               self._previous_blocks_completed.split(",")]

    def update_blocks(self):
        """It creates a list of blocks completed by the user with the first
        block in the list being the one the user most recently completed. It
        also rewrites the updated history of the user to the file."""
        self.get_previous_block_completion()
        if self._completion_state == "true":
            self._previous_blocks_completed = [block for block in
                                               self._previous_blocks_completed
                                               if block != self._block_completed]
            self._previous_blocks_completed.insert(0, self._block_completed)
            self._updated_blocks = self._previous_blocks_completed
            self.df_completed.loc[self.df_completed["UserId"] == self._user_id,
                                  "completed_blocks"] = ", ".join(self._updated_blocks)

            self.df_completed.to_csv(self.completed_blocks_filepath, index=False)

    def match_topics(self):
        self.update_blocks()
        with open(self.json_keys_file) as json_file:
            json_keys = json.load(json_file)
        conf_topics = json_keys["sh_rec"]["Topics"]
        topics_mapped = \
            self.conf_cms_file[self.conf_cms_file["Conf_topic"].isin(conf_topics)]

        return topics_mapped

    def create_sh_input(self):
        topics_mapped = self.match_topics()
        matched_blocks = [b for b in self._updated_blocks if b in topics_mapped["monitor"].values]

        if len(self._updated_blocks) > len(matched_blocks):
            logger.warning(
                "The following blocks are not mapped in the config file: {}".format(
                    set(self._updated_blocks) - set(matched_blocks)))

        self._completed_topics = topics_mapped.set_index("monitor").loc[matched_blocks, "Conf_topic"].to_list()

        if self._completed_topics:
            topic_numbers = ["topic-" + str(i) for i in
                             range(1, len(self._completed_topics) + 1)]
            input_dict = dict(zip(topic_numbers, self._completed_topics))
            user_id_dict = {"idx": self._user_id}
            sh_input = {**user_id_dict, **input_dict}
            return pd.DataFrame([sh_input])
        else:
            # print("Adding a random topic since no matched topics.")
            random_topic = random.choice([*topics_mapped['Conf_topic'].unique()])
            input_dict = {"topic-1": random_topic}
            user_id_dict = {"idx": self._user_id}
            sh_input = {**user_id_dict, **input_dict}
            return pd.DataFrame(sh_input, index=[0])


class Recommender:
    """
    Args: The same as CreateSHInput class with the addition of the
        sh_pickle_path (the complete path with the filename mentioned in it of
        the pickle out of the sh_habtonomic class).

    Returns: The recommended blocks for each decider type by mapping them from
        the conf_cms file. It also formats the output as required for delivery.
    """
    DECIDER_CODE_MAPPING = {"reinforced": "203",
                            "related": "202",
                            "new": "201",
                            "reinforced_linear": "213",
                            "related_linear": "212",
                            "new_linear": "211",
                            "reinforced_multiply": "223",
                            "related_multiply": "222",
                            "new_multiply": "221",
                            }  # mapping of decider types

    SBUS_USER_ID_KEY = "UserId"
    SBUS_BLOCK_CODE_KEY = "BlockCode"
    SBUS_EVENT_ID_KEY = "Id"

    def __init__(self, service_bus_dict: dict,
                 base_path: str = pkg_resources.resource_filename("sh_recommender", "data"),
                 completed_blocks_filename: str = "blocks_completed.csv",
                 conf_cms_filename: str = "conf_cms_topics_mapping.csv",
                 keys_filename: str = "keys.json",
                 sh_pickle_path: str = "sh_habtonomics.pickle",
                 ):

        self._base_path = base_path
        self._conf_cms_filename = conf_cms_filename
        self._sb_message = service_bus_dict
        with open(os.path.join(base_path, sh_pickle_path), "rb") as pckl:
            self.sh = pickle.load(pckl)

        self.sh_input_creation = SHInput(base_path,
                                         completed_blocks_filename,
                                         conf_cms_filename, keys_filename,
                                         service_bus_dict)
        self.sh_input = None
        self.sh_decider = Decider(shadow_habtonomics=self.sh)
        self.max_block_score = 10  # To order the recommednations made

        self.conf_cms_file = pd.read_csv(os.path.join(self._base_path,
                                                      self._conf_cms_filename))


    def get_decider_rec(self):
        # outputs the recommendations of the decider for each type as a list
        self.sh_input = self.sh_input_creation.create_sh_input()
        if self.sh_input is not None:
            final_rec_topics = self.sh_decider.predict(self.sh_input)
            return (final_rec_topics["reinforced"][0],
                    final_rec_topics["related"][0],
                    final_rec_topics["new"][0],
                    final_rec_topics["reinforced_linear"][0],
                    final_rec_topics["related_linear"][0],
                    final_rec_topics["new_linear"][0],
                    final_rec_topics["reinforced_multiply"][0],
                    final_rec_topics["related_multiply"][0],
                    final_rec_topics["new_multiply"][0],
                    )
        else:
            return None, None, None, None, None, None, None, None, None

    def map_topics_to_blocks(self, topic_list):
        """This maps the topics recommended to the first knowledge blocks"""
        try:
            ds_nowledge_blocks = self.conf_cms_file \
                .drop_duplicates(subset="Conf_topic") \
                .set_index("Conf_topic")["first_knowledge_block_code"]

            l_knowledge_blocks = ds_nowledge_blocks[topic_list].tolist()

            l_knowledge_blocks  = l_knowledge_blocks

            return l_knowledge_blocks

        except KeyError:
            logger.warning("At least one of these topics is not in the conf_cms file:" + str(topic_list))
            return None

    def get_knowledge_blocks_rec(self):
        rec_topics_reinforced, rec_topics_related, rec_topics_new, \
            rec_topics_reinforced_linear, rec_topics_related_linear, rec_topics_new_linear, \
            rec_topics_reinforced_multiply, rec_topics_related_multiply, rec_topics_new_multiply = \
            self.get_decider_rec()

        return (self.map_topics_to_blocks(rec_topics_reinforced),
                self.map_topics_to_blocks(rec_topics_related),
                self.map_topics_to_blocks(rec_topics_new),
                self.map_topics_to_blocks(rec_topics_reinforced_linear),
                self.map_topics_to_blocks(rec_topics_related_linear),
                self.map_topics_to_blocks(rec_topics_new_linear),
                self.map_topics_to_blocks(rec_topics_reinforced_multiply),
                self.map_topics_to_blocks(rec_topics_related_multiply),
                self.map_topics_to_blocks(rec_topics_new_multiply),
                )

    def create_block_priorities(self, topic_list):
        # Orders the blocks recommended with an artificially created block score
        block_priorities = []
        try:
            for i in range(len(topic_list)):
                temp_dict = {"BlockCode": topic_list[i],
                             "BlockScore": (self.max_block_score - i) / self.max_block_score}
                block_priorities.append(temp_dict)
            return block_priorities
        except:
            return None

    def create_json_message(self, topic_list, decidertype):
        # creates a json message for each decider type
        block_priorities = self.create_block_priorities(topic_list)
        final_json = {"UserId": self._sb_message[Recommender.SBUS_USER_ID_KEY],
                      "TimestampUtc": str(datetime.utcnow()),
                      "DeciderType": str(decidertype),
                      "DecisionId": str(uuid.uuid4()),
                      "BlockPriorities": block_priorities}

        return final_json

    def create_all_json_messages(self):
        # Outputs the json messages, one for each decider type
        rec_topics_reinforced, rec_topics_related, rec_topics_new, \
            rec_topics_reinforced_linear, rec_topics_related_linear, rec_topics_new_linear, \
            rec_topics_reinforced_multiply, rec_topics_related_multiply, rec_topics_new_multiply, = \
            self.get_knowledge_blocks_rec()

        reinforced_json = self.create_json_message(rec_topics_reinforced,
                                                   Recommender.DECIDER_CODE_MAPPING["reinforced"])

        reinforced_linear_json = self.create_json_message(rec_topics_reinforced_linear,
                                                   Recommender.DECIDER_CODE_MAPPING["reinforced_linear"])

        reinforced_multiply_json = self.create_json_message(rec_topics_reinforced_multiply,
                                                   Recommender.DECIDER_CODE_MAPPING["reinforced_multiply"])

        related_json = self.create_json_message(rec_topics_related,
                                                Recommender.DECIDER_CODE_MAPPING["related"])

        related_linear_json = self.create_json_message(rec_topics_related_linear,
                                                          Recommender.DECIDER_CODE_MAPPING["related_linear"])

        related_multiply_json = self.create_json_message(rec_topics_related_multiply,
                                                            Recommender.DECIDER_CODE_MAPPING["related_multiply"])

        new_json = self.create_json_message(rec_topics_new,
                                            Recommender.DECIDER_CODE_MAPPING["new"])

        new_linear_json = self.create_json_message(rec_topics_new_linear,
                                                          Recommender.DECIDER_CODE_MAPPING["new_linear"])

        new_multiply_json = self.create_json_message(rec_topics_new_multiply,
                                                            Recommender.DECIDER_CODE_MAPPING["new_multiply"])

        #block_predictor_json = self.create_json_message(rec_topics_block_predictor, Recommender.DECIDER_CODE_MAPPING["block_predictor"])
        return (json.dumps(reinforced_json),
                json.dumps(related_json),
                json.dumps(new_json),
                json.dumps(reinforced_linear_json),
                json.dumps(related_linear_json),
                json.dumps(new_linear_json),
                json.dumps(reinforced_multiply_json),
                json.dumps(related_multiply_json),
                json.dumps(new_multiply_json),
                )
