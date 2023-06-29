import requests
import pandas as pd
from pathlib import Path


# Download schedule data from https://www.airportal.go.kr/life/airinfo/abc.SheetAction
def download_schedule(start, end, arrival=True):
    url = 'https://www.airportal.go.kr/life/airinfo/abc.SheetAction'
    if not type(start) == pd.Timestamp:
        start = pd.to_datetime(start)
    if not type(end) == pd.Timestamp:
        end = pd.to_datetime(end)

    periods = []
    # Get last day of the month
    while start < end:
        if start.month == 12:
            period = [start, start.replace(month=12, day=31)]
            start = start.replace(month=12, day=31) + pd.Timedelta(days=1)
        else:
            # Get last day of the month
            period = [start, start.replace(day=1) + pd.offsets.MonthEnd(1)]
            start = start.replace(month=start.month + 1, day=1)
        if period[1] > end:
            period[1] = end
        periods.append([period[0].strftime('%Y%m%d'), period[1].strftime('%Y%m%d')])

    dfs = []

    for period in periods:
        # Check for cache file
        filename = 'data/airportal/%s_%s_%s.csv' % ('arrival' if arrival else 'departure', period[0], period[1])

        try:
            df = pd.read_csv(filename)
            dfs.append(df)
            continue
        except:
            pass

        print('Downloading %s to %s' % (period[0], period[1]))
        data = {
            'S_CONTROLLER': 'aipsPack.service.FlightScheduleService',
            'S_METHOD': 'searchArrToExcel' if arrival else 'searchDepToExcel',
            'S_SAVENAME': 'GUBUN|SCH_DATE|AL_ICAO|FP_IATA|AP_DEP_IATA|AP_ICAO_KR|AP_ARR_IATA|AP_ARR_KR|SCH_TIME|ETD|ATD|NAT|STATUS',
            'S_FORWARD': '',
            'S_HEADSTR': '',
            'ad_gubun': 'A' if arrival else 'D',
            'sDate': '%s' % period[0],
            'eDate': '%s' % period[1],
            'airport': 'RKSI',
            'ibTabTop0': '',
            'editpage0': '',
            'ibTabBottom0': ''
        }
        response = requests.post(url, data=data)
        # Decode korean characters and read xml
        raw = response.content.decode('euc-kr')
        # Replace <DATA *> to <table>
        # Remove text before <DATA
        raw = raw[raw.find('<DATA '):]
        raw = raw.replace('<DATA ', '<table ')
        raw = raw.replace('</DATA>', '</table>')
        raw = raw.replace('<![CDATA[', '')
        raw = raw.replace(']]>', '')
        df = pd.read_html(raw)[0]
        dfs.append(df)
        # Save cache file
        # Create directory if not exists
        Path('data/airportal').mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df.columns = ['GUBUN', 'SCH_DATE', 'AL_ICAO', 'FP_IATA', 'AP_DEP_IATA', 'AP_ICAO_KR', 'AP_ARR_IATA', 'AP_ARR_KR',
                  'SCH_TIME', 'ETD', 'ATD', 'NAT', 'STATUS']
    return df
