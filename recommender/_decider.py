# Databricks notebook source
import math
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from sh_recommender._sh import ShadowHabtonomics
from sh_recommender.block_predictor import scoring_table

logger = logging.getLogger(__package__)


def _cycle_list(l, n):
    """Cycle n elements of a list from the ending to the begining"""
    return l[-n:] + l[:-n]


class UserPreferences:
    """This class is responsible for generating the user's preferences based on
    user's historic
    """

    def __init__(
            self,
            shadow_habtonomics: ShadowHabtonomics,
            n_min_historic: int = 1,
            user_id_col: str = "idx",
    ):
        """UserPreferences constructor

        Args:
            shadow_habtonomics (ShadowHabtonomics): The Shadow Habtonomics
            n_min_historic (int, optional): Minimum size of historic for calculating user preferences.
                If the size of historic is smaller than n_min_historic, the returned preferences will be uniform.
                Defaults to 2.
            user_id_col (str, optional): User identification column name. Defaults to "idx".
        """

        self.shadow_habtonomics = shadow_habtonomics
        self.n_min_historic = n_min_historic
        self.user_id_col = user_id_col

    def decay_function(self, user_historic):
        """This is the function that is applied to the historic to give more importance to more recent finished topics.

        Args:
            user_historic (pd.DataFrame): Users historics as DataFrame.

        Returns:
            List[float]: List of user preferences for each Shadow Habtonomic.
        """
        result = {}
        weight = 1.0
        total = 0.0
        for i in range(1, len(user_historic) + 1):
            sh = user_historic.loc[f"topic-{i}"]
            if math.isnan(sh):
                break
            total += 1 / i
            result[sh] = result.get(sh, 0.0) + 1 / i
        for sh in result.keys():
            result[sh] = result[sh] / total

        return result

    @property
    def topic2sh(self):
        """Get a dict mapping from topic to Shadow Habtonomic.

        Returns:
            Dict: Topics mapped to Shadow Habtonomic.
        """

        topics = self.shadow_habtonomics.nodes_table
        topic2sh = dict(list(zip(topics.topic, topics.shadow_habtonomics)))

        return topic2sh

    @property
    def n_sh(self):
        """Get the number of Shadow Habtonomics.

        Returns:
            int: Number of unique Shadow Habtonomics.
        """

        return len(self.shadow_habtonomics.nodes_table.shadow_habtonomics.unique())

    def get_user_preferences(self, users_historic):
        """Get user preferences for each Shadow Habtonomic based on historic.

        Args:
            users_historic (pd.DataFrame): Users historics as DataFrame.

        Returns:
            pd.DataFrame: Users' preferences for each Shadow Habtonomic.
        """

        # decay_function = self.decay_function
        n_min_historic = self.n_min_historic
        user_id_col = self.user_id_col
        topic2sh = self.topic2sh
        n_sh = self.n_sh

        users_historic = users_historic.set_index(user_id_col)
        for c in users_historic.columns:
            users_historic[c] = users_historic[c].map(topic2sh)
        users_historic = users_historic.T

        users = users_historic.columns
        historic_size = users_historic.notnull().sum().to_dict()
        user_preference = np.zeros((len(users), n_sh))

        for i, c in enumerate(users):
            if historic_size[c] >= n_min_historic:
                sh_probabilities = self.decay_function(users_historic[c])
                for item, value in sh_probabilities.items():
                    user_preference[i, int(item)] = value
            else:
                user_preference[i, :] = 1 / n_sh

        return pd.DataFrame(user_preference, index=users, columns=[f"SH_{i}" for i in range(n_sh)])


class Reward:
    """This class is responsible for the calculation of reward vectors associated with topics.
    """

    def __init__(self, shadow_habtonomics: ShadowHabtonomics):
        """Reward class constructor.

        Args:
            shadow_habtonomics (ShadowHabtonomics): The Shadow Habtonomics network.
        """

        self.shadow_habtonomics = shadow_habtonomics

    @property
    def topics(self):
        """List the unique topics in Shadow Habtonomics network.

        Returns:
            List[str]: The list of unique topics.
        """

        return self.shadow_habtonomics.nodes_table.topic.unique()

    @property
    def topic2sh(self):
        """Get a dict mapping from topic to Shadow Habtonomic.

        Returns:
            Dict: Topics mapped to Shadow Habtonomic.
        """

        topics = self.shadow_habtonomics.nodes_table
        topic2sh = dict(list(zip(topics.topic, topics.shadow_habtonomics)))

        return topic2sh

    @property
    def n_sh(self):
        """Get the number of Shadow Habtonomics.

        Returns:
            int: Number of unique Shadow Habtonomics.
        """

        return len(self.shadow_habtonomics.nodes_table.shadow_habtonomics.unique())

    @property
    def topics2neighbors(self):
        """Dict mapping from topics to its neighbors in the Shadow Habtonomics network.

        Returns:
            Dict: Dict mapping from topics to its neighbors in the Shadow Habtonomics network.
        """

        topics2neighbors = {}

        for topic in self.topics:
            topics2neighbors[topic] = [
                e[1]
                for e in self.shadow_habtonomics.shadow_habtonomics_network.edges(topic)
            ]

        return topics2neighbors

    def get_rewards(self):
        """Get rewards for each topic.

        Returns:
            pd.DataFrame: Pandas DataFrame containing the reward vectors for each topic.
        """

        rewards = np.zeros((len(self.topics), self.n_sh))

        for i, topic in enumerate(self.topics):

            # The SH to which this topic belongs has a reward of 1.0
            rewards[i, self.topic2sh[topic]] = 1.0

            topic_neighbors = self.topics2neighbors[topic]

            # For all other SHs, the reward is the average weight of the connection to all neighbors in that SH
            sh_rewards = {}
            for neighbor in topic_neighbors:
                neighbor_sh = self.topic2sh[neighbor]

                # We already assigned the reward for this topic's SH
                if self.topic2sh[topic] == neighbor_sh:
                    continue

                # Accumulate the weights
                sh_rewards[neighbor_sh] = sh_rewards.get(neighbor_sh, []) + [
                    self.shadow_habtonomics.shadow_habtonomics_network[topic][neighbor][
                        "weight"
                    ]
                ]

            # Normalize and assign to the reward matrix
            for item, value in sh_rewards.items():
                sh_reward_mean = sum(value) / len(value)
                rewards[i, item] = sh_reward_mean

        rewards = pd.DataFrame(
            rewards, index=self.topics, columns=[f"SH_{i}" for i in range(self.n_sh)]
        )

        return rewards


class Decider(BaseEstimator, ClassifierMixin):
    """The Shadow Habtonomics recommenders class.
    """
    CHOICES_PER_DECIDER = 3
    available_types = ["reinforced", "related", "new",
                       "reinforced_linear", "related_linear", "new_linear",
                       "reinforced_multiply", "related_multiply", "new_multiply"]
    def __init__(
            self,
            shadow_habtonomics: ShadowHabtonomics,
            up_n_min_historic: int = 1,
            user_id_col: str = "idx",
    ):
        """Decider Constructor

        Args:
            shadow_habtonomics (ShadowHabtonomics): The Shadow Habtonomics
            n_min_historic (int, optional): Minimum size of historic for calculating user preferences.
                If the size of historic is smaller than n_min_historic, the returned preferences will be uniform.
                Defaults to 2.
            user_id_col (str, optional): User identification column name. Defaults to "idx".
        """

        self.shadow_habtonomics = shadow_habtonomics
        self.n_topics: int = len(self.shadow_habtonomics.nodes_table.topic.unique())
        self.up_n_min_historic = up_n_min_historic
        self.user_id_col = user_id_col
        self.user_preferences_calculator = UserPreferences(
            shadow_habtonomics,
            n_min_historic=up_n_min_historic,
            user_id_col=user_id_col,
        )

        self.reward_calculator = Reward(shadow_habtonomics)
        self.predictor_score = scoring_table

    def fit(self, X: pd.DataFrame = None, y=None):
        """This method is here for standard purposes.

        Args:
            X (pd.DataFrame, optional): Not used. Defaults to None.
            y ([type], optional): Not used. Defaults to None.

        Returns:
            Decider: Return the Decider (self).
        """

        return self

    def fit_predict_each_decider_type(self, X: pd.DataFrame, decider_type):
        """Run a specific decider and get its recommendations for the users historics.

        Args:
            X (pd.DataFrame): Users historic.
            decider_type (str): Which decider should be run.

        Returns:
            np.array: Array containing the recommendations for each user.
        """
        epsilon = 0.005
        users_preferences = self.user_preferences_calculator.get_user_preferences(X)
        reward_vectors = self.reward_calculator.get_rewards()
        topics2neighbors = self.reward_calculator.topics2neighbors
        topic2sh = self.reward_calculator.topic2sh
        user_id_col = self.user_id_col

        user_history_df = X.copy()
        user_history_df = user_history_df.set_index(user_id_col)

        # Load scores with Neighborhood Score Calculation
        scores = pd.DataFrame(
            users_preferences.values @ reward_vectors.T.values,
            index=users_preferences.index,
            columns=reward_vectors.index,
        )
        logger.debug(str(decider_type) + " scores before history is applied:")
        logger.debug(scores.T.sample(frac=1))

        for user in users_preferences.index:
            #print("score: \n", block_finish_score)
            df_topic_scores = pd.DataFrame(
                np.NAN,  # cause otherwise dtype will be object
                index=reward_vectors.index,
                columns=["SHP_score", "n_completed_neighbours"])

            user_historic = user_history_df.loc[user].drop_duplicates().to_list()
            block_finish_score = pd.DataFrame()
            # get user score
            if user in list(self.predictor_score["UserId"].unique()):
                block_finish_score = self.predictor_score[self.predictor_score["UserId"] == user].reset_index(drop=True) \
                                                        [["next_time_rate", "topic"]].set_index('topic').sort_index()
            for topic in reward_vectors.index:
                neighbors = topics2neighbors[topic]

                # get sh of topics of interest which are this + it's neighbors
                l_topics_sh = ["SH_" + str(topic2sh[topic]) for topic in [topic] + neighbors]

                # SHP is the mean of the preference scores for these blocks
                df_topic_scores.loc[topic, "SHP_score"] = users_preferences.loc[user, l_topics_sh].mean()

                # get number of completed neighbors
                df_topic_scores.loc[topic, "n_completed_neighbours"] = len(
                    set(neighbors).intersection(set(user_historic)))

            # in order to avoid overlapping topics, we separate them into 2 groups: low and high SHP
            n_els = df_topic_scores.shape[0] // 2
            assert n_els > self.CHOICES_PER_DECIDER, "Not enough topics to choose from."

            # need to sort by topic as well to break ties
            l_sorted_topics = df_topic_scores.reset_index().sort_values(by=["SHP_score", "index"])["index"].to_list()
            shp_low, shp_high = l_sorted_topics[:n_els], l_sorted_topics[n_els:]

            # Main formula to calculate scores
            scores.loc[user] *= df_topic_scores["n_completed_neighbours"]

            if decider_type == "reinforced":
                scores.loc[user] *= df_topic_scores["SHP_score"]

                # in the reinforced recommender scores for topics with low SHP are set to -2
                scores.loc[user, shp_low] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

            elif decider_type == "reinforced_linear":
                scores.loc[user] *= df_topic_scores["SHP_score"]

                # in the reinforced recommender scores for topics with low SHP are set to -2
                scores.loc[user, shp_low] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

                count_score = scores.transpose()[user].value_counts()
                # adjust SH score
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores.loc[user] = scores.loc[user] - scores.loc[user].min() + epsilon
                # combining SH score vs probability
                if user in list(self.predictor_score["UserId"].unique()):
                    scores.loc[user] = 0.3 * scores.loc[user] + 0.7 * (block_finish_score["next_time_rate"].transpose())

            elif decider_type == "reinforced_multiply":
                scores.loc[user] *= df_topic_scores["SHP_score"]

                # in the reinforced recommender scores for topics with low SHP are set to -2
                scores.loc[user, shp_low] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

                # adjust SH score
                count_score = scores.transpose()[user].value_counts()
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores.loc[user] = scores.loc[user] - scores.loc[user].min() + epsilon
                # multiply SH score vs probability
                if user in list(self.predictor_score["UserId"].unique()):
                    scores.loc[user] *= block_finish_score["next_time_rate"]

            elif decider_type == "related":
                scores.loc[user] *= (1 - df_topic_scores["SHP_score"])
                # in the related recommender scores for topics with high SHP are set to -2
                scores.loc[user, shp_high] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

            elif decider_type == "related_linear":
                scores.loc[user] *= (1 - df_topic_scores["SHP_score"])
                # in the reinforced recommender scores for topics with low SHP are set to -2
                scores.loc[user, shp_low] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

                count_score = scores.transpose()[user].value_counts()
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores.loc[user] = scores.loc[user] - scores.loc[user].min() + epsilon

                if user in list(self.predictor_score["UserId"].unique()):
                    scores.loc[user] = 0.3 * scores.loc[user] + 0.7 * (block_finish_score["next_time_rate"].transpose())

            elif decider_type == "related_multiply":
                scores.loc[user] *= (1 - df_topic_scores["SHP_score"])
                # in the reinforced recommender scores for topics with low SHP are set to -2
                scores.loc[user, shp_low] = -2
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)

                count_score = scores.transpose()[user].value_counts()
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores.loc[user] = scores.loc[user] - scores.loc[user].min() + epsilon

                if user in list(self.predictor_score["UserId"].unique()):
                    scores.loc[user] *= block_finish_score["next_time_rate"]

            # ## edge cases
            # scores for topics without completed neighbors are set to 0 automatically in the multiplication above
            # topics already played get a score in the range [-2,-1] the longer ago it was played the lower the score
            else:
                scores.loc[user, user_historic] = np.arange(-2, -1, 1 / len(user_historic)) + 1 / len(user_historic)
        print("score: ", scores)
        scores = scores.T.sample(frac=1)
        logger.debug(str(decider_type) + " scores after history is applied:")
        logger.debug(scores)
        best_choices = []

        for user in scores.columns:
            user_historic = list(user_history_df.loc[user])
            logger.debug("user_historic: " + str(user_historic))
            block_finish_score = self.predictor_score[self.predictor_score["UserId"] == user].reset_index(drop=True)
            block_finish_score = block_finish_score[["next_time_rate", "topic"]].set_index('topic').sort_index()

            if decider_type == "new":
                # for the new recommender every block that wasn't played has the same probability of being chosen
                scores.loc[scores[user] > 0, user] = 0

            elif decider_type == "new_linear":
                scores.loc[scores[user] > 0, user] = 0

                count_score = scores[user].value_counts()
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores[user] = scores[user] - scores[user].min() + epsilon

                if user in list(self.predictor_score["UserId"].unique()):
                    scores[user] = 0.3 * scores[user] + 0.7 * (block_finish_score["next_time_rate"])

            elif decider_type == "new_multiply":
                scores.loc[scores[user] > 0, user] = 0

                count_score = scores[user].value_counts()
                if 0 in count_score:
                    if count_score[0] >= 6:
                        scores[user] = scores[user] - scores[user].min() + epsilon

                if user in list(self.predictor_score["UserId"].unique()):
                    scores[user] *= (block_finish_score["next_time_rate"].transpose())

            best_choices.append(scores[user].nlargest(self.CHOICES_PER_DECIDER).index.to_list())
            logger.debug("New best_choices for recommender " + str(decider_type) + ":")
            logger.debug(str(best_choices) + "\n")
        return best_choices

    def fit_predict(self, X: pd.DataFrame, y=None):
        """Run all deciders and get its recommendations for the users historics.

        Args:
            X (pd.DataFrame): Users historics.
            y: Ignored.

        Returns:
            pd.DataFrame: DataFrame containing the recommendations for each user.
        """

        logger.debug("Predict called with X input:")
        logger.debug(X)

        results = {}
        # adjust probability bases on topics finished priorities
        user_history_df = X.copy().set_index(self.user_id_col)
        for user in user_history_df.index:
            user_historic = user_history_df.loc[user].dropna().to_list()
            divided_char = len(user_historic)
            for topic in user_historic:
                temp = divided_char/(user_historic.index(topic)+1)
                self.predictor_score.loc[((self.predictor_score["UserId"] == user) &
                                      (self.predictor_score["topic"] == topic)), "next_time_rate"] /= temp
        for decider_type in Decider.available_types:
            results[decider_type] = self.fit_predict_each_decider_type(X, decider_type)

        results[self.user_id_col] = X[self.user_id_col]

        return pd.DataFrame(results)

    def predict(self, X: pd.DataFrame, y=None):
        """Same as fit_predict"""
        return self.fit_predict(X)
