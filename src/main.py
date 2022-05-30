import sys, requests, json, dateutil
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from gekko import GEKKO

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib.ticker as plticker

from onboard.client import ProductionAPIClient
from onboard.client.dataframes import points_df_from_timeseries
from onboard.client.models import PointSelector, TimeseriesQuery, PointData

from decouple import config


rc('font',**{'family':'serif','serif':['Times New Roman'], 'size': 10})

def usage():
    print("Usage:")
    print("python time-series.py <building name or id> "
          "<equipment type name + suffix> <start date> <end date>")
    print("e.g. python time-series.py \"Science Park\" \"AHU SSAC-1\" 2019-10-15 2019-10-31")
    sys.exit(1)

def optimal_cp(demand_array, TOU, curtailment = True):
    m = GEKKO(remote=False)
    m.time = np.linspace(0,23,24)
    demand_max = max(demand_array)
    demand_max = demand_max if demand_max % 100 == 0 else demand_max + 100 - demand_max % 100
    cap_low = [demand_max-d for d in demand_array]
    cap_up = 24*[demand_max * 0.8] if max(cap_low) < demand_max * 0.9 else 24*[demand_max]
    cap_low = cap_low if max(cap_low) < max(cap_up) else [i*0.8 for i in cap_low]
    k = 1.1 #Safety factor
    CL = m.Param(value=cap_low)
    CH = m.Param(value=cap_up)
    TOU = m.Param(value=TOU)
    if curtailment == True:
      x = m.MV(lb=3, ub=6.6)
      x.STATUS = 1
    else:
      x = 6.6
    n = m.MV(lb=1, ub = max(cap_low)/6 if max(cap_low)%6!=0 else max(cap_low)/5.95, integer=True)
    n.STATUS = 1
    C = m.SV(value=cap_low[0])
    m.Equations([C>=CL, C<=CH])
    m.Equation(k*C - n* x == 0)
    m.Minimize(TOU*n*x)
    m.options.IMODE = 6
    try:
      m.solve(disp=False,debug=False)
      return m, C, n, x, cap_low, cap_up
    except:
      optimal_cp(demand_array, TOU, curtailment = False)
      return m, C, n, x, cap_low, cap_up
    else:
      print(e)
      return 6 * [None]

def date_preparation(df, bulding_name, point_id, print_image = False):
    df['datetime'] = pd.to_datetime(df['time'])
    try:
      df = df.drop(['time', 'clean'], axis=1)
    except Exception as e:
      print(e)
      pass
    df['Month'] = df['datetime'].dt.month
    df['Year'] = df['datetime'].dt.year
    df['Week'] = df['datetime'].dt.isocalendar().week
    Monthly = pd.pivot_table(df,index='Month',columns='Year',values='raw',aggfunc=np.sum)
    Weekly = pd.pivot_table(df,index='Week',columns='Year',values='raw',aggfunc=np.sum)
    df = df.set_index('datetime')
    df = df.replace('?', np.nan)
    df = df.astype(np.float64).fillna(method='bfill')
    df_hourly = df.resample('H').mean()
    df_daily = df.resample('D').sum()
    df_daily_m = df.resample('D').mean()
    df_daily['Date'] = df_daily.index
    df_daily ['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['Weekday'] = df_daily['Date'].dt.day_name()
    df_daily = df_daily.drop(['Date'], axis=1)
    df_weekly_sum = df.resample('W-MON').sum()
    df_weekly_mean = df.resample('W-MON').mean()
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly.index = df_hourly.index.date
    df_pivot = df_hourly.pivot(columns='hour')
    df_pivot = df_pivot.dropna()
    df_pivot_power = df_pivot['raw'].copy()
    if print_image:
      ax1 = df_pivot_power.T.plot(figsize=(6,4), legend=False, color='black', alpha=0.2)
      ax1.set_xlabel(r"$Time$ (hr)")
      ax1.set_ylabel(r"$Consumption$ (kW)")
      plt.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.13)
      plt.savefig(str(bulding_name) + '_' + str(point_id) +'_consumption')
      plt.clf()
    return df_pivot_power

def find_silhouette(df_pivot_power, bulding_name, point_id, print_image = False):
    sillhoute_scores = []
    n_cluster_list = np.arange(2,31).astype(int)
    X = df_pivot_power.values.copy()
    # Very important to scale!
    sc = MinMaxScaler()
    fit_transform = sc.fit_transform(X)
    for n_cluster in n_cluster_list:
      kmeans = KMeans(n_clusters=n_cluster)
      cluster_found = kmeans.fit_predict(X)
      sillhoute_scores.append(silhouette_score(fit_transform, kmeans.labels_))
    if print_image:
      plt.plot(sillhoute_scores, 'k')
      plt.ylabel(r'$Silhouette$')
      plt.xlabel(r'$Cluster$\#')
      plt.title('Silhouette Analysis')
      plt.grid(True)
      plt.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.13)
      plt.savefig(str(bulding_name) + '_' + str(point_id) +'_sillhoute')
      plt.clf()
    return fit_transform

def find_k_means(df_pivot_power, fit_transform, bulding_name, bulding_type, point_id, TOU, curtailment = True, print_image = False):
    kmeans = KMeans(n_clusters = 5)
    cluster_found = kmeans.fit_predict(fit_transform)
    cluster_found_sr = pd.Series(cluster_found, name='cluster')
    df_pivot_power = df_pivot_power.set_index(cluster_found_sr, append=True)
    cluster_values = sorted(df_pivot_power.index.get_level_values('cluster').unique())
    baseloads = [(list(df_pivot_power.xs(cluster, level=1).median(numeric_only=True))[:24]) for cluster in cluster_values]
    baseload_max = [max(baseload) for baseload in baseloads]
    baseload_cluster = baseload_max.index(max(baseload_max))
    baseload = list(df_pivot_power.xs(baseload_cluster, level=1).median(numeric_only=True))[:24]
    m, C, n, x, cap_low, cap_up = optimal_cp(baseload, TOU, curtailment)

    if m and C and n and x and cap_low and cap_up:
      accumulated_power = [a*b for a,b in zip(list(n.value),list(x.value))] if curtailment else len(m.time)*[x * min(n.value[1:])]
      accumulated_cost = sum([a*b for a,b in zip(TOU,accumulated_power)])/100.0
      if print_image:
        fig = plt.figure(figsize=(6.4, 9.2))
        gs  = gridspec.GridSpec(4, 1, height_ratios=[2, 1 ,1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])
        color_list = ['blue', 'red', 'green', 'black', 'gray']
        for cluster, color in zip(cluster_values, color_list):
          df_pivot_power.xs(cluster, level=1).T.plot(ax=ax1, legend=False, alpha=0.01, color=color, label= False)
          df_pivot_power.xs(cluster, level=1).median().plot(ax=ax1, color=color, legend=False, label= False, alpha= 1, ls='--' )
        ax1.set_title(f'Load Profile {bulding_name}_{point_id}',fontsize=10) 
        ax1.set_ylabel(r'Consumption (kW)',fontsize=10)
        lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='--') for c in color_list]
        labels = [f'cluster#{cluster+1}' for cluster in cluster_values]
        ax1.legend(lines, labels)
        ax1.xaxis.set_visible(False)
        ax1.set_xticklabels([])
        ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax1.grid('on', which='major', axis='x' )
        ax2.plot(m.time, cap_up,'g-')
        ax2.plot(m.time, cap_low, 'k--')
        ax2.plot(m.time,C.value,'r:')
        ax2.set_title('Capacity',fontsize=10)
        ax2.set_ylim([1,int(max(cap_up)*1.2)])
        ax2.set_ylabel('Power (kW)',fontsize=10)
        ax2.set_xticklabels([])
        ax2.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax2.legend(["Upper bound", "Lower bound", "Asgd power"])

        ax3.step(m.time,x.value,'b--') if curtailment else ax3.step(m.time,len(m.time)* [x],'b--')
        ax3.set_title('Chargers Schedule Setpoint',fontsize=10)
        ax3.set_ylabel('Power (kW)',fontsize=10)
        ax3.set_ylim([1,int(max(x.value)*1.2)]) if curtailment else ax3.set_ylim([1,x*1.2])
        ax3.set_xticklabels([])
        ax3.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))

        ax4.step(m.time,n.value,'r-') if curtailment else ax4.step(m.time,len(m.time)* min(n.value[1:]),'r-')
        ax4.set_title('No. of Chargers',fontsize=10)
        ax4.set_ylim([1,int(max(n.value)*1.2)])
        ax4.set_xlabel('Daytime',fontsize=10)
        ax4.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax4.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        fig.tight_layout()
        plt.savefig(f'Profile {bulding_type} {bulding_name} {point_id}')
        plt.clf()
##        plt.close(fig)
      return baseload, max(n.value), min(n.value[1:]), accumulated_cost
    else:
      return baseload, 0, 0

def get_building_id(client, building):
    try:
        return int(building)
    except ValueError:
        pass
    buildings = client.get_all_buildings()
    matches = [b.get('id') for b in buildings if b.get('name') == building]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Found multiple buildings named {building} - "
              f"ids = {matches} - please retry using an id")
        return None
    return matches[0]

def get_equipment(client, building_id, equip_type_id):
    all_equipment = client.get_building_equipment(building_id)
    for e in all_equipment:
        if e['equip_type_id'] == equip_type_id:
            return e
    return None

def save_building_table(cp_numbers, curtailment):
    df = pd.DataFrame(data_points)
    df = df.drop(['first_updated', 'last_updated', 'name'], axis=1)
    df.to_csv('building_info_slms.csv', sep=',', encoding='utf-8') if curtailment else df.to_csv('building_info_lms.csv', sep=',', encoding='utf-8')

def fetch_time_series(client, building_id, equip_type_id, start, end):
    equipment = get_equipment(client, building_id, equip_type_id)
    if equipment is None:
        print(f"Could not find equipment with suffix '{equip_suffix}'")
        usage()
    point_ids = [p['id'] for p in equipment['points']]
    timeseries_query = TimeseriesQuery(point_ids = point_ids, start = start, end = end)
    query_results = client.stream_point_timeseries(timeseries_query)
    return points_df_from_timeseries(query_results)

def building_info(client, building_ids = None):
  if api_key is None:
    print("API key must be set as environment variable ONBOARD_API_KEY")
    sys.exit(1)
  data_poits = []
  bdgs_with_panel = []
  key = { "key": api_key}
  response = requests.post("https://api.onboarddata.io/login/api-key", data=key).json()
  headers = {"Authorization": "Bearer "+ response["access_token"]}
  if building_ids == None:
    bdgs = requests.get("https://api.onboarddata.io/buildings", headers=headers).json()
    building_ids = [pdg.get('id') for pdg in bdgs]

  for bdgs_id in building_ids:
    try:
      res = requests.get(f"https://api.onboarddata.io/buildings/{bdgs_id}", headers=headers)
      if res.status_code == 200:
        res = res.json()
        bdg_type = res.get('info').get('customerType')
        point_count = res.get('point_count')
        equip_count = res.get('equip_count')
        area = res.get('sq_ft')
        query = PointSelector()
        query.equipment_types = ['meter']
        query.buildings = [bdgs_id]
        selection = client.select_points(query)
        points = selection["points"]
        if points:
          bdgs_with_panel.append(bdgs_id)
          for point in points:        
            res = requests.get(f"https://api.onboarddata.io/buildings/{bdgs_id}/points/{point}", headers=headers).json()
            if res['measurement_id'] in [13, 14]:
              data_poits.append({'building': bdgs_id,
                                 'type':bdg_type,
                                 'point': point,
                                 'first_updated': res['first_updated'],
                                 'last_updated': res['last_updated'],
                                 'description': res['description'],
                                 'name': res['name'],
                                 'area': area,
                                 'equip_count': equip_count,
                                 'point_count': point_count})
    except Exception as e:
      print(e)
      pass
  return data_poits, bdgs_with_panel

def select_ToU(date, building_type):
  start_tier_1 = dateutil.parser.parse(config('ST_TIER_1')).replace(year=datetime.now().year)
  start_tier_2 = dateutil.parser.parse(config('ST_TIER_2')).replace(year=datetime.now().year)
  start_tier_3 = dateutil.parser.parse(config('ST_TIER_3')).replace(year=datetime.now().year)

  if start_tier_1 < date < start_tier_2:
    if building_type in ['multifamily', 'residential']:
      return json.loads(config('RESIDENTIAL_TOU_1'))
    else:
      return json.loads(config('BUSINESS_TOU_1'))
  if start_tier_2 < date < start_tier_3:
    if building_type in ['multifamily', 'residential']:
      return json.loads(config('RESIDENTIAL_TOU_2'))
    else:
      return json.loads(config('BUSINESS_TOU_2'))
  else:
    if building_type in ['multifamily', 'residential']:
      return json.loads(config('RESIDENTIAL_TOU_3'))
    else:
      return json.loads(config('BUSINESS_TOU_3'))

if __name__ == '__main__':
  try:
    bdgs_ids = json.loads(config('BUILDING_IDs'))
  except:
    bdgs_ids = None
  api_key = config('ONBOARD_API_KEY')
  save_to_csv = config('SAVE_CSV', default=False, cast=bool)
  curtailment = config('CURTAILMENT', default=True, cast=bool)

  date_now = datetime.now()
  client = ProductionAPIClient(api_key=api_key)
  data_points, bdgs_with_panel = building_info(client, bdgs_ids)
  if data_points != []:
    for data in data_points:
      building_id = data['building']
      building_type = data['type'].lower() if data['type'] else 'commercial'
      building_name = data['name']
      point_id = data['point']
      TOU = select_ToU(date_now, building_type)
      try:
        if (any(x in building_name.lower() for x in ['building kw', 'commercial retail']) or
            (building_type in ['multifamily'] and not any(x in building_name.lower() for x in ['gas', 'kwh', 'electric']))):
          timeseries_query = TimeseriesQuery(point_ids = [point_id],
                                             start = datetime.fromtimestamp(data["first_updated"]/1000, timezone.utc),
                                             end = datetime.fromtimestamp(data["last_updated"]/1000, timezone.utc))
          query_results = client.stream_point_timeseries(timeseries_query)
          for point in query_results:
            df = pd.DataFrame(point.values)
            df.columns = point.columns          
            df_pivot_power = date_preparation(df, building_id, point_id)
            fit_transform = find_silhouette(df_pivot_power, building_id, point_id)
            baseload, max_cp, min_cp, accumulated_cost = find_k_means(df_pivot_power,
                                                                      fit_transform,
                                                                      building_id,
                                                                      building_type,
                                                                      point_id, TOU,
                                                                      curtailment,
                                                                      print_image = True)
            if save_to_csv:
              df.to_csv(f'B_{building_id}_P_{point_id}_{building_type}_{building_name.replace(" ","_").replace(":","_").replace(".","_").replace(chr(92),"_").replace("/","_")}.csv',
                        sep=',',
                        encoding='utf-8')
            if baseload and curtailment:
              data.update({'max_demand': max(baseload), 'Charger_num': int(max_cp), 'Cost ($/day)': accumulated_cost})
            elif baseload and not curtailment:
              data.update({'max_demand': max(baseload), 'Charger_num': int(min_cp), 'Cost ($/day)': accumulated_cost})
            else:
              data.update({'max_demand': None, 'Charger_num': None, 'Cost ($/day)': None})
        else:
          data.update({'max_demand': None, 'Charger_num': None, 'Cost ($/day)': None})
      except Exception as e:
        print(e)
        pass
    save_building_table(data_points, curtailment)
  else:
    print(f'No data is recorded for builidng/s {bdgs_ids}')

 

