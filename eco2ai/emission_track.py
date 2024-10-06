import os
import time
import platform
import pandas as pd
import numpy as np
import uuid
import warnings
import tzlocal
from apscheduler.schedulers.background import BackgroundScheduler

from eco2ai.tools.tools_gpu import GPU, all_available_gpu
from eco2ai.tools.tools_cpu import CPU, all_available_cpu
from eco2ai.tools.tools_ram import RAM
from eco2ai.utils import (
    is_file_opened,
    define_carbon_index,
    get_params,
    set_params,
    # calculate_money,
    # summary,
    encode,
    encode_dataframe,
    electricity_pricing_check,
    calculate_price,
    FileDoesNotExistsError,
    NotNeededExtensionError,
)

FROM_mWATTS_TO_kWATTH = 1000 * 1000 * 3600
FROM_kWATTH_TO_MWATTH = 1000


class IncorrectMethodSequenceError ( Exception ):
    pass


class Tracker:


    def __init__(
            self,
            project_name=None,
            experiment_description=None,
            file_name=None,
            measure_period=10,
            emission_level=None,
            alpha_2_code=None,
            region=None,
            cpu_processes="current",
            pue=1,
            encode_file=None,
            electricity_pricing=None,
            ignore_warnings=False,
    ):

        self._ignore_warnings = ignore_warnings
        if not self._ignore_warnings:
            warnings.warn (
                message=""
            )
        if (type ( measure_period ) == int or type ( measure_period ) == float) and measure_period <= 0:
            raise ValueError ( "\'measure_period\' should be positive number" )
        if encode_file is not None:
            if type ( encode_file ) is not str and not (encode_file is True):
                raise TypeError ( f"'encode_file' parameter should have str type, not {type ( encode_file )}" )
            if type ( encode_file ) is str and not encode_file.endswith ( '.csv' ):
                raise NotNeededExtensionError ( f"'encode_file' name need to be with extension \'.csv\'" )
        if file_name is not None:
            if type ( file_name ) is not str and not (file_name is True):
                raise TypeError ( f"'file_name' parameter should have str type, not {type ( file_name )}" )
            if type ( file_name ) is str and not file_name.endswith ( '.csv' ):
                raise NotNeededExtensionError ( f"'file_name' name need to be with extension \'.csv\'" )
        self._params_dict = get_params ()
        self.project_name = project_name if project_name is not None else self._params_dict["project_name"]
        self.experiment_description = experiment_description if experiment_description is not None else \
        self._params_dict["experiment_description"]
        self.file_name = file_name if file_name is not None else self._params_dict["file_name"]
        self._measure_period = measure_period if measure_period is not None else self._params_dict["measure_period"]
        self._pue = pue if pue is not None else self._params_dict["pue"]
        self.get_set_params ( self.project_name, self.experiment_description, self.file_name, self._measure_period,
                              self._pue )

        self._emission_level, self._country = define_carbon_index ( emission_level, alpha_2_code, region )
        self._cpu_processes = cpu_processes
        self._scheduler = BackgroundScheduler (
            job_defaults={'max_instances': 10},
            timezone=str ( tzlocal.get_localzone () ),
            misfire_grace_time=None
        )
        self._start_time = None
        self._cpu = None
        self._gpu = None
        self._ram = None
        self._id = None
        self._current_epoch = "N/A"
        self._consumption = 0
        self._encode_file = encode_file if encode_file != True else "encoded_" + file_name
        electricity_pricing_check ( electricity_pricing )
        self._electricity_pricing = electricity_pricing
        self._total_price = 0
        self._os = platform.system ()
        if self._os == "Darwin":
            self._os = "MacOS"
        # self._mode == "first_time" means that the Tracker is just initialized
        # self._mode == "run time" means that CO2 tracker is now running
        # self._mode == "shut down" means that CO2 tracker is stopped
        # self._mode == "training" means that CO2 tracker tracks training process
        self._mode = "first_time"
        # parameters to save during model training
        self._parameters_to_save = ""

    def get_set_params(
            self,
            project_name=None,
            experiment_description=None,
            file_name=None,
            measure_period=None,
            pue=None
    ):

        dictionary = dict ()
        if project_name is not None:
            dictionary["project_name"] = project_name
        else:
            dictionary["project_name"] = "default project name"
        if experiment_description is not None:
            dictionary["experiment_description"] = experiment_description
        else:
            dictionary["experiment_description"] = "default experiment description"
        if file_name is not None:
            dictionary["file_name"] = file_name
        else:
            dictionary["file_name"] = "emission.csv"
        if measure_period is not None:
            dictionary["measure_period"] = measure_period
        else:
            dictionary["measure_period"] = 10
        if pue is not None:
            dictionary["pue"] = pue
        else:
            dictionary["pue"] = 1
        set_params ( **dictionary )

        return dictionary

    def consumption(self):

        return self._consumption

    def price(self):
        return self._total_price

    def id(self):

        return self._id

    def emission_level(self):

        return self._emission_level

    def measure_period(self):

        return self._measure_period

    def _construct_attributes_dict(self, ):

        # if user used older versions, it may be needed to upgrade his .csv file
        # but after all, such verification should be deleted
        # self.check_for_older_versions()
        attributes_dict = dict ()
        attributes_dict["id"] = [self._id]
        attributes_dict["project_name"] = [f"{self.project_name}"]
        attributes_dict["experiment_description"] = [f"{self.experiment_description}"]
        attributes_dict["epoch"] = [
            "epoch: " + str ( self._current_epoch ) + self._parameters_to_save if self._mode == "training" else "N/A"
        ]
        attributes_dict["start_time"] = [
            f"{time.strftime ( '%Y-%m-%d %H:%M:%S', time.localtime ( self._start_time ) )}"]
        attributes_dict["duration(s)"] = [f"{time.time () - self._start_time}"]
        attributes_dict["power_consumption(kWh)"] = [f"{self._consumption}"]
        attributes_dict["CO2_emissions(kg)"] = [f"{self._consumption * self._emission_level / FROM_kWATTH_TO_MWATTH}"]
        attributes_dict["CPU_name"] = [f"{self._cpu.name ()}/{self._cpu.cpu_num ()} device(s), TDP:{self._cpu.tdp ()}"]
        attributes_dict["GPU_name"] = [f"{self._gpu.name ()} {self._gpu.gpu_num ()} device(s)"]
        attributes_dict["OS"] = [f"{self._os}"]
        attributes_dict["region/country"] = [f"{self._country}"]
        attributes_dict["cost"] = [f"{self._total_price}"]

        return attributes_dict

    def _write_to_csv(
            self,
            add_new=False,
    ):
        # if user used older versions, it may be needed to upgrade his .csv file
        # but after all, such verification should be deleted
        # self.check_for_older_versions()
        attributes_dict = self._construct_attributes_dict ()

        if not os.path.isfile ( self.file_name ):
            while True:
                if not is_file_opened ( self.file_name ):
                    open ( self.file_name, "w" ).close ()
                    tmp = open ( self.file_name, "w" )
                    pd.DataFrame ( attributes_dict ).to_csv ( self.file_name, index=False )
                    tmp.close ()
                    break
                else:
                    time.sleep ( 0.5 )

        else:
            while True:
                if not is_file_opened ( self.file_name ):
                    # we open a file in order to the current system process know that this file is opened somewhere
                    # Thus, if it's open in the process, other processes won't open it, until it will be closed
                    tmp = open ( self.file_name, "r" )

                    attributes_dataframe = pd.read_csv ( self.file_name )
                    attributes_array = []
                    for element in attributes_dict.values ():
                        attributes_array += element

                    if attributes_dataframe[attributes_dataframe['id'] == self._id].shape[0] == 0:
                        attributes_dataframe.loc[attributes_dataframe.shape[0]] = attributes_array
                    else:
                        row_index = attributes_dataframe[attributes_dataframe['id'] == self._id].index.values[-1]
                        # check, if it's necessary to add a new row to the dataframe
                        if add_new:
                            attributes_dataframe = pd.DataFrame (
                                np.vstack ( (
                                    attributes_dataframe.values[:row_index + 1],
                                    attributes_array,
                                    attributes_dataframe.values[row_index + 1:]
                                ) ),
                                columns=attributes_dataframe.columns
                            )
                        else:
                            attributes_dataframe.loc[row_index] = attributes_array
                    attributes_dataframe.to_csv ( self.file_name, index=False )

                    tmp.close ()
                    break
                else:
                    time.sleep ( 0.5 )

        self._mode = "run time" if self._mode != "training" else "training"
        return attributes_dict

    def _update_to_new_version(self, attributes_dataframe, new_columns):
        current_columns = list ( attributes_dataframe.columns )
        for column in new_columns:
            if column not in current_columns:
                attributes_dataframe[column] = "N/A"
        attributes_dataframe = attributes_dataframe[new_columns]

        return attributes_dataframe

    def _func_for_sched(self, add_new=False):

        cpu_consumption = self._cpu.calculate_consumption ()
        ram_consumption = self._ram.calculate_consumption ()
        if self._gpu.is_gpu_available:
            gpu_consumption = self._gpu.calculate_consumption ()
        else:
            gpu_consumption = 0
        tmp_comsumption = 0
        tmp_comsumption += cpu_consumption
        tmp_comsumption += gpu_consumption
        tmp_comsumption += ram_consumption
        tmp_comsumption *= self._pue
        if self._electricity_pricing is not None:
            self._total_price += calculate_price ( self._electricity_pricing, tmp_comsumption )
        self._consumption += tmp_comsumption

        # self._consumption = 0
        # self._start_time = time.time()
        if self._mode == "shut down":
            self._scheduler.remove_job ( "job" )
            self._scheduler.shutdown ()
        # self._write_to_csv returns attributes_dict
        return self._write_to_csv ( add_new )

    def start_training(self, start_epoch=1):

        if not isinstance ( start_epoch, int ):
            raise TypeError (
                f"\"start_epoch\" paramenet must be of int type. Now, it is {type ( start_epoch )}"
            )

        self._mode = "training"

        self._current_epoch = start_epoch
        self._cpu = CPU ( cpu_processes=self._cpu_processes, ignore_warnings=self._ignore_warnings )
        self._gpu = GPU ( ignore_warnings=self._ignore_warnings )
        self._ram = RAM ( ignore_warnings=self._ignore_warnings )
        self._id = str ( uuid.uuid4 () )
        self._start_time = time.time ()

    def new_epoch(self, parameters_dict):
        if self._mode != "training":
            raise IncorrectMethodSequenceError (
                "You can run method \".new_epoch\" only after method \".start_training\" was run"
            )
        self._parameters_to_save = ", "
        for key in parameters_dict:
            self._parameters_to_save += key + ": "
            self._parameters_to_save += str ( parameters_dict[key] ) + ", "
        # self._func_for_sched returns attributes_dict.
        attributes_dict = self._func_for_sched ( add_new=True )
        # We put it into self._func_for_encoding method in order to encode calculations
        if self._encode_file:
            self._func_for_encoding ( attributes_dict )
        self._current_epoch += 1
        self._parameters_to_save = ""
        self._consumption = 0
        self._total_price = 0
        self._start_time = time.time ()
        if self._encode_file is not None:
            self._func_for_encoding ( attributes_dict )
        self._consumption = 0

    def start(self):

        if self._mode == "training":
            raise IncorrectMethodSequenceError (
                """
You have already run ".start_training" method.
Please, use the interface for training: ".start_trainig", ".new_epoch", and "stop_training"
                """
            )
        if self._start_time is not None:
            try:
                self._scheduler.remove_job ( "job" )
                self._scheduler.shutdown ()
            except:
                pass
            self._scheduler = BackgroundScheduler ( job_defaults={'max_instances': 10}, misfire_grace_time=None )
        self._cpu = CPU ( cpu_processes=self._cpu_processes, ignore_warnings=self._ignore_warnings )
        self._gpu = GPU ( ignore_warnings=self._ignore_warnings )
        self._ram = RAM ( ignore_warnings=self._ignore_warnings )
        self._id = str ( uuid.uuid4 () )
        self._mode = "first_time"
        self._start_time = time.time ()
        self._scheduler.add_job ( self._func_for_sched, "interval", seconds=self._measure_period, id="job" )
        self._scheduler.start ()

    def stop_training(self, ):

        # remove job from scheduler
        if self._mode != "training" or self._start_time is None:
            raise IncorrectMethodSequenceError (
                """
You should run ".start_training" method before ".stop_training" method
                """
            )
        self._consumption = 0
        self._mode = "shut down"

    def stop(self, ):
        if self._mode == "training":
            self.stop_training ()
            return
        if self._start_time is None:
            raise Exception ( "Need to first start the tracker by running tracker.start() or tracker.start_training()" )
        self._scheduler.remove_job ( "job" )
        self._scheduler.shutdown ()
        self._func_for_sched ()
        attributes_dict = self._write_to_csv ()
        if self._encode_file is not None:
            self._func_for_encoding ( attributes_dict )
        self._start_time = None
        self._consumption = 0
        self._mode = "shut down"

    def _func_for_encoding(self, attributes_dict):
        for key in attributes_dict.keys ():
            # attributes_dict[key] = [encode(str(attributes_dict[key][0]))]
            attributes_dict[key] = [encode ( str ( value ) ) for value in attributes_dict[key]]

        if not os.path.isfile ( self._encode_file ):
            while True:
                if not is_file_opened ( self._encode_file ):
                    open ( self._encode_file, "w" ).close ()
                    tmp = open ( self._encode_file, "r" )
                    pd.DataFrame ( attributes_dict ).to_csv ( self._encode_file, index=False )

                    tmp.close ()
                    break
                else:
                    time.sleep ( 0.5 )

        else:
            while True:
                if not is_file_opened ( self._encode_file ):
                    tmp = open ( self._encode_file, "r" )

                    attributes_dataframe = pd.read_csv ( self._encode_file )

                    attributes_dataframe = pd.concat (
                        [
                            attributes_dataframe,
                            pd.DataFrame ( attributes_dict ),
                        ],
                        ignore_index=True,
                        axis=0
                    )

                    attributes_dataframe.to_csv ( self._encode_file, index=False )
                    tmp.close ()
                    break
                else:
                    time.sleep ( 0.5 )
def track(func):

    def inner(*args, **kwargs):
        tracker = Tracker()
        tracker.start()
        try:
            returned = func(*args, **kwargs)
        except Exception:
            tracker.stop()
            del tracker
            raise Exception
        tracker.stop()
        del tracker
        return returned

    return inner


# Apply the track decorator to your existing function
@track
def my_function():
    for _ in range(10):
        _ = _ + 1
        # ... (rest of your code)

# Call the decorated function
my_function()



