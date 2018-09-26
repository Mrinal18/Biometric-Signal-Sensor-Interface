from hexoskin import client, errors
import utility_helper as util
import csv, time, calendar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


"""
    4145 : Acc X,
    4146 : Acc Y,
    4147 : Acc Z
    49 : act1s,
    50 : act15s,
    51 : act5m
    16 : ecg
    19 : hr
    36 : mv (minute ventilation),
    37 : vt (ventilation time)
    33 : respiration rate r_rr
    4129 : resp thoracic, 4130 : resp abdominal
    1000 : HR-quality, 1001, BR-quality, 1002: spo2-quality
"""
def get_record_list(auth, limit='100', user='', device_filter=''):
    filters = {}
    if limit != '100':
        filters['limit'] = limit
    if user != '':
        filters['user'] = user
    if device_filter != '':
        filters['device'] = device_filter
    records = auth.api.record.list(filters)
    return records.response.result['objects']

def get_record_data(record_id):
    data = auth.api.data.list(record=record_id, datatype=4113, flat=True)
    datalist = [row for row in data]
    df = pd.DataFrame(datalist)
    df.columns = ["time", "heart"]
    return df

def get_active_record_list(auth, limit='100', user='', device_filter=''):
    record_list = get_record_list(auth, limit, user, device_filter)
    response = []
    for record in record_list:
        if record['status'] == 'complete':
            response.append(record['id'])
    return response

def realtime_data_generator(auth, record_id, datatypes):
    record = auth.api.record.get(record_id)
    start_time = (calendar.timegm(time.gmtime()) - 20) * 256
    end_time = (calendar.timegm(time.gmtime()) - 10) * 256

    while True:
        user = record.user
        data = get_realtime_data_helper(auth, user, start_time, end_time, datatypes) # why not sending record
        record = auth.api.record.get(record_id)
        start_time = end_time
        end_time = (calendar.timegm(time.gmtime()) - 10) * 256
        time.sleep(6)
        yield data

    return

def get_realtime_data_helper(auth, user, start, end, datatypes):
    final_data = {}
    for data_id in datatypes:
        # this is for checking if raw data is needed or not
        # raw_flag = False
        # key = [data_key for data_key, data_value in util.datatype.items() if data_value == [data_id]]
        # if !len(key):
        #     key = [data_key for data_key, data_value in util.raw_datatype.items() if data_value == [data_id]]
        #     raw_flag = True

        data = get_data_helper(auth, user, start, end, data_id)
        final_data[data_id] = data

    return final_data

def get_data_helper(auth, user, start, end, data_id):
    data_sample_rate = util.data_sample_rate[data_id]
    if data_sample_rate is not None:



def main(argv):
    auth = util.api_login()
    record_id = get_active_record_list(auth)[0]
    datatypes = [4113]
    while True:
        realtime_data_generator(auth, record_id, datatypes)

if __name__ == '__main__':
    main(sys.argv)
