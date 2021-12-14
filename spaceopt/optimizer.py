import random
from typing import Dict, List, Union
import lightgbm as lgb
import pandas as pd
from spaceopt.space import Space


class SpaceOpt:

    _ALLOWED_OBJECTIVE_VALUES = ("maximize", "minimize", "max", "min")
    _FIT_PREDICT_MIN_EVALUATED_SPOINTS = 2

    def __init__(
        self,
        search_space: Dict[str, list],
        target_name: str,
        objective: str,
    ) -> None:
        self.space = Space(search_space=search_space)
        self._verify_target_name(target_name)
        self.target_name = target_name
        self._verify_objective(objective)
        self.objective = objective
        self.evaluated_spoints = []
        self.lgb_params = {
            "objective": "mae",
            "learning_rate": 0.1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "num_threads": 1,
            "verbose": -1,
        }

    def append_evaluated_spoint(self, evaluated_spoint: dict) -> None:
        self._verify_evaluated_spoint(evaluated_spoint)
        self.evaluated_spoints.append(evaluated_spoint)

    def get_random(
        self,
        num_spoints: int = 1,
        sample_size: int = 10000,
    ) -> Union[dict, List[dict]]:
        self._verify_num_spoints(num_spoints)
        self._verify_sample_size(num_spoints)
        spoints = self._sample_unevaluated_unique_spoints(
            sample_size=max(sample_size, num_spoints),
        )
        spoints = spoints[:num_spoints]
        if num_spoints == 1:
            return spoints[0]
        return spoints

    def fit_predict(
        self,
        num_spoints: int = 1,
        num_boost_round: int = 1024,
        sample_size: int = 10000,
    ) -> Union[dict, List[dict]]:
        self._verify_num_spoints(num_spoints)
        self._verify_num_boost_round(num_boost_round)
        self._verify_sample_size(num_spoints)
        if len(self.evaluated_spoints) < self._FIT_PREDICT_MIN_EVALUATED_SPOINTS:
            return self.get_random(num_spoints=num_spoints, sample_size=sample_size)
        evaluated_spoints_df = pd.DataFrame(self.evaluated_spoints)
        evaluated_spoints_df = self.space.encode_variables(evaluated_spoints_df)
        dataset = lgb.Dataset(
            data=evaluated_spoints_df[self.space.variable_names],
            label=evaluated_spoints_df[self.target_name],
            categorical_feature=self.space.categorical_names,
        )
        model = lgb.train(
            params=self.lgb_params,
            train_set=dataset,
            valid_sets=[dataset],
            num_boost_round=num_boost_round,
            categorical_feature=self.space.categorical_names,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )
        spoints = self._sample_unevaluated_unique_spoints(sample_size=sample_size)
        spoints_df = pd.DataFrame(spoints)
        spoints_df = self.space.encode_variables(spoints_df)
        spoints_df[self.target_name] = model.predict(
            spoints_df[self.space.variable_names]
        )
        spoints_df = spoints_df.sort_values(
            by=self.target_name,
            ascending=self._is_ascending_sorting(),
        )
        spoints_df = self.space.decode_variables(spoints_df)
        spoints_df = spoints_df[self.space.variable_names]
        spoints_df = spoints_df.iloc[:num_spoints]
        spoints = spoints_df.to_dict("records")
        if num_spoints == 1:
            return spoints[0]
        return spoints

    def _is_ascending_sorting(self) -> bool:
        if self.objective in ("minimize", "min"):
            return True
        elif self.objective in ("maximize", "max"):
            return False
        else:
            raise ValueError(f"unknown objective={self.objective}")

    def _sample_random_spoints(self, sample_size: int) -> List[dict]:
        self._verify_sample_size(sample_size)
        return [self.space.sample() for i in range(sample_size)]

    def _sample_unevaluated_unique_spoints(
        self,
        sample_size: int,
        max_num_retries: int = 100,
    ) -> List[dict]:
        self._verify_sample_size(sample_size)
        self._verify_max_num_retries(max_num_retries)
        if len(self.evaluated_spoints) > 0:
            evaluated_spoints_df = pd.DataFrame(self.evaluated_spoints)
            evaluated_spoints_df = evaluated_spoints_df[self.space.variable_names]
            evaluated_spoints_df = evaluated_spoints_df.drop_duplicates()
            num_unique_evaluated_spoints = len(evaluated_spoints_df)
        else:
            num_unique_evaluated_spoints = 0
        max_sample_size = min(
            sample_size, self.space.size - num_unique_evaluated_spoints
        )
        sampled_spoints = []
        for i in range(max_num_retries):
            sampled_spoints += self._sample_random_spoints(sample_size=sample_size)
            sampled_spoints_df = pd.DataFrame(sampled_spoints)
            sampled_spoints_df = sampled_spoints_df.drop_duplicates()
            if num_unique_evaluated_spoints > 0:
                sampled_spoints_df = sampled_spoints_df.merge(
                    right=evaluated_spoints_df,
                    on=self.space.variable_names,
                    how="left",
                    indicator=True,
                )
                indicator_left_only = sampled_spoints_df["_merge"].eq("left_only")
                sampled_spoints_df = sampled_spoints_df[indicator_left_only]
                sampled_spoints_df = sampled_spoints_df.drop(columns="_merge")
            sampled_spoints = sampled_spoints_df.to_dict("records")
            if len(sampled_spoints) >= max_sample_size:
                break
        sampled_spoints = sampled_spoints[:max_sample_size]
        if len(sampled_spoints) == 0:
            raise RuntimeError(
                "could not sample any new spoints -"
                " search_space is fully explored or random sampling was unfortunate."
                f"\nsearch_space.size = {self.space.size}"
                f"\nnum evaluated spoints = {len(self.evaluated_spoints)}"
                f"\nnum unevaluated spoints = {self.space.size - len(self.evaluated_spoints)}"
            )
        random.shuffle(sampled_spoints)
        return sampled_spoints

    def _verify_evaluated_spoint(self, evaluated_spoint: dict) -> None:
        if not isinstance(evaluated_spoint, dict):
            raise TypeError(
                f"evaluated_spoint is of type {type(evaluated_spoint)},"
                f" but it should be of type {dict}."
            )
        self.space.verify_spoint(evaluated_spoint)
        if self.target_name not in evaluated_spoint:
            raise ValueError(
                f"spoint={evaluated_spoint} is not evaluated,"
                f" target_name={repr(self.target_name)} is not found."
            )
        if not isinstance(evaluated_spoint[self.target_name], float):
            raise TypeError(
                f"evaluated_spoint has {repr(self.target_name)}"
                f" with value={evaluated_spoint[self.target_name]}"
                f" of type {type(evaluated_spoint[self.target_name])},"
                f" but it should be of type {float}."
            )

    def _verify_max_num_retries(self, max_num_retries: int) -> None:
        if not isinstance(max_num_retries, int):
            raise TypeError(
                f"max_num_retries is of type {type(max_num_retries)},"
                f" but it should be of type {int}."
            )
        if not max_num_retries > 0:
            raise ValueError("max_num_retries should be greater than 0.")

    def _verify_num_boost_round(self, num_boost_round: int) -> None:
        if not isinstance(num_boost_round, int):
            raise TypeError(
                f"num_boost_round is of type {type(num_boost_round)},"
                f" but it should be of type {int}."
            )
        if not num_boost_round > 0:
            raise ValueError("num_boost_round should be greater than 0.")

    def _verify_num_spoints(self, num_spoints: int) -> None:
        if not isinstance(num_spoints, int):
            raise TypeError(
                f"num_spoints is of type {type(num_spoints)},"
                f" but it should be of type {int}."
            )
        if not num_spoints > 0:
            raise ValueError("num_spoints should be greater than 0.")

    def _verify_objective(self, objective: str) -> None:
        if objective not in self._ALLOWED_OBJECTIVE_VALUES:
            raise ValueError(
                f"objective should be one of: {self._ALLOWED_OBJECTIVE_VALUES}."
            )

    def _verify_sample_size(self, sample_size: int) -> None:
        if not isinstance(sample_size, int):
            raise TypeError(
                f"sample_size is of type {type(sample_size)},"
                f" but it should be of type {int}."
            )
        if not sample_size > 0:
            raise ValueError("sample_size should be greater than 0.")

    def _verify_target_name(self, target_name: str) -> None:
        if not isinstance(target_name, str):
            raise TypeError(
                f"target_name is of type {type(target_name)},"
                f" but it should be of type {str}."
            )
        if target_name == "":
            raise ValueError("target_name is empty.")
        if target_name in self.space.variable_names:
            raise RuntimeError(
                f"target_name={repr(target_name)} should not be"
                f" in search space variables: {self.space.variable_names}."
            )

    def __str__(self) -> str:
        indent = " " * 4
        innerstr = []
        innerstr += [str(self.space).replace("\n", "\n" + indent)]
        innerstr += [f"target_name={repr(self.target_name)}"]
        innerstr += [f"objective={self.objective}"]
        innerstr = indent + (",\n" + indent).join(innerstr)
        outstr = "{cls}(\n{innerstr}\n)".format(
            cls=self.__class__.__name__,
            innerstr=innerstr,
        )
        return outstr
