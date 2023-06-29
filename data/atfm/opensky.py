import paramiko
import pandas as pd

from .utils import multiprocessing
from tqdm import tqdm


# Table list
#                         name
# 0                 acas_data4
# 1      allcall_replies_data4
# 2                  flarm_raw
# 3                    flights
# 4              flights_data4
# 5       identification_data4
# 6   operational_status_data4
# 7             position_data4
# 8     rollcall_replies_data4
# 9          sensor_visibility
# 10   sensor_visibility_data3
# 11             state_vectors
# 12       state_vectors_data3
# 13       state_vectors_data4
# 14            velocity_data4

# state_vectors_data4 table description
#              name        type                      comment
# 0            time         int  Inferred from Parquet file.
# 1          icao24      string  Inferred from Parquet file.
# 2             lat      double  Inferred from Parquet file.
# 3             lon      double  Inferred from Parquet file.
# 4        velocity      double  Inferred from Parquet file.
# 5         heading      double  Inferred from Parquet file.
# 6        vertrate      double  Inferred from Parquet file.
# 7        callsign      string  Inferred from Parquet file.
# 8        onground     boolean  Inferred from Parquet file.
# 9           alert     boolean  Inferred from Parquet file.
# 10            spi     boolean  Inferred from Parquet file.
# 11         squawk      string  Inferred from Parquet file.
# 12   baroaltitude      double  Inferred from Parquet file.
# 13    geoaltitude      double  Inferred from Parquet file.
# 14  lastposupdate      double  Inferred from Parquet file.
# 15    lastcontact      double  Inferred from Parquet file.
# 16        serials  array<int>  Inferred from Parquet file.
# 17           hour         int                         None

class OpenSky:
    def __init__(self, username, password, num_workers=1, verbose=False):
        def connect_to_server():
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, self.port, self.username, self.password)
            return client
        self.host = 'data.opensky-network.org'
        self.port = 2230
        self.username = username
        self.password = password
        self.clients = []
        self.verbose = verbose
        if self.verbose:
            print('Connecting to %s:%d...' % (self.host, self.port))
            pbar = tqdm(total=num_workers)
        for i in range(num_workers):
            client = self.connect()
            self.clients.append(client)

            if self.verbose:
                pbar.update(1)
        if self.verbose:
            pbar.close()
            print('Connected to %s:%d!' % (self.host, self.port))

        self.state_vectors_data4_columns = ['time', 'icao24', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'callsign', 'onground', 'alert', 'spi', 'squawk', 'baroaltitude', 'geoaltitude', 'lastposupdate', 'lastcontact', 'serials', 'hour']

    def connect(self, in_use=False):
        client = paramiko.SSHClient()
        client.in_use = in_use
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, self.port, self.username, self.password)
        return client

    def get(self, command):
        for i, client in enumerate(self.clients):
            if not client.in_use:
                client.in_use = True
                try:
                    stdin, stdout, stderr = client.exec_command(command)
                    lines = stdout.readlines()
                    client.in_use = False
                    return lines
                # If connection is lost, reconnect and try again
                except paramiko.SSHException:
                    if self.verbose:
                        print('Connection lost, reconnecting...')
                    client.close()
                    self.clients[i] = self.connect()
                    return self.get(command)


    def close(self):
        for client in self.clients:
            client.close()

    def query(self, query):
        if self.verbose:
            print('Querying %s...' % query)
        data = self.get('-q %s' % query)
        return parse_data(data)

    def show_tables(self):
        return self.query('show tables;')

    def describe_table(self, table):
        return self.query('describe %s;' % table)

    def get_vectors(self, start_time, end_time, callsign=None, enroute_only=True):
        # check if columns are valid

        start_hour = int(start_time // 3600) * 3600
        end_hour = int(end_time // 3600) * 3600

        select_query = ('select ceiling(lastposupdate) as time, '
                        'avg(lat) as lat, '
                        'avg(lon) as lon, '
                        'avg(velocity) as velocity, '
                        'avg(heading) as heading, '
                        'avg(vertrate) as vertrate, '
                        'avg(geoaltitude) as geoaltitude, '
                        'avg(baroaltitude) as baroaltitude ')
        from_query = 'from state_vectors_data4 '
        where_query = ('where time <= %d and time >= %d '
                       'and hour <= %d and hour >= %d '
                       % (end_time, start_time, end_hour, start_hour))
        if callsign is not None:
            where_query += 'and trim(callsign)="%s" ' % callsign
        if enroute_only:
            where_query += 'and onground=0 '
        group_query = 'group by ceiling(lastposupdate);'

        if self.verbose:
            print('Querying flight %s between %d and %d...' % (callsign, start_time, end_time))
        df = self.query(select_query + from_query + where_query + group_query)
        if self.verbose:
            print('Downloaded %s vectors for flight %s!' % (len(df), callsign))
        return df

    def get_vectors_by_callsigns(self, payloads):
        def get_vectors(payload):
            start_time, end_time, callsign = payload
            return self.get_vectors(start_time, end_time, callsign)

        # Payloads contains a list of tuples (start_time, end_time, callsign)
        # Download using multiple threads
        results = multiprocessing(get_vectors, payloads, num_thread=len(self.clients))
        return results

        # Combine results



def parse_data(data):
    headers = []
    rows = []
    for row in data:
        cols = strip_row(row)
        if len(cols) == 0:
            continue
        elif len(cols) == 1 and cols[0][:2] == '+-':
            continue
        if len(headers) == 0:
            headers = cols
        else:
            rows.append(cols)

    return pd.DataFrame(rows, columns=headers)


def strip_row(row):
    cols = [item.strip() for item in row.split('|')]
    while '' in cols:
        cols.remove('')
    # Convert 'NULL' to None
    for i, col in enumerate(cols):
        if col == 'NULL':
            cols[i] = None
    return cols


if __name__ == '__main__':
    from datetime import datetime

    opensky = OpenSky('joshuad', 'aelics070', num_workers=3, verbose=True)
    payload = (
        (1672511160, 1672514760, 'PAC226'),
        (1672496580, 1672500180, 'MAA3955'),
        (1672531320, 1672534920, 'GTI545')
    )
    vectors = opensky.get_vectors_by_callsigns(payload)
