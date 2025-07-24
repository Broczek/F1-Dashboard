import os
import asyncio
import re
import base64
from io import BytesIO
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fastf1
import fastf1.plotting
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

fastf1.plotting.setup_mpl(color_scheme='fastf1', misc_mpl_mods=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

def clean_text(text):
    return re.sub(r'\[\d+\]', '', text).strip()

def load_session_data(year: int, location: str, session_identifier: str):
    try:
        session = fastf1.get_session(year, location, session_identifier)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        if session.laps is None or session.laps.empty:
            return None
        return session
    except Exception as e:
        print(f"Error while loading session {year} {location} {session_identifier}: {e}")
        return None

@app.get("/")
async def read_root():
    return {"message": "Welcome to the F1 dashboard API! The server is operational"}

def scrape_teams_data_sync(year: int):
    base_url = "https://www.formula1.com"
    teams_url = f"{base_url}/en/teams"
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    
    try:
        print(f"Rozpoczynanie scrapingu strony głównej zespołów: {teams_url}")
        main_response = requests.get(teams_url, headers=headers)
        main_response.raise_for_status()
        
        soup = BeautifulSoup(main_response.content, 'html.parser')
        team_items = soup.select("a.group\\/team-card")
        print(f"{len(team_items)} potential team members found on the home page")
        
        all_teams_data = []

        for item in team_items:
            team_data = {}
            
            name_tag = item.select_one("p.typography-module_display-l-bold__m1yaJ")
            
            if not name_tag:
                continue

            team_data['name'] = name_tag.text.strip()

            logo_tag = item.select_one(".TeamLogo-module_teamlogo__lA3j1 img")
            if logo_tag and logo_tag.has_attr('src'):
                team_data['logoUrl'] = re.sub(r'w_\d+', 'w_800', logo_tag['src'])
            else:
                team_data['logoUrl'] = None

            car_tag = item.select_one("span.relative.h-\\[112px\\] img")
            if car_tag and car_tag.has_attr('src'):
                raw_url = car_tag['src']
                high_quality_url = re.sub(r'(/upload/).*?(/v\d+/)', r'\1\2', raw_url)
                team_data['carUrl'] = high_quality_url
            else:
                team_data['carUrl'] = None


            detail_url = f"{base_url}{item['href']}"
            print(f"Scraping the detailed page for {team_data['name']}: {detail_url}")

            try:
                detail_response = requests.get(detail_url, headers=headers)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
                
                stats_containers = detail_soup.select("dl.DataGrid-module_dataGrid__Zk5Y8")
                for container in stats_containers:
                    stats_items = container.select("div.DataGrid-module_item__cs9Zd")
                    for stat_item in stats_items:
                        title_tag = stat_item.find("dt")
                        value_tag = stat_item.find("dd")
                        if title_tag and value_tag:
                            title = title_tag.text.strip().lower()
                            value = value_tag.text.strip()
                            
                            if 'full team name' in title: team_data['fullName'] = value
                            elif 'team chief' in title: team_data['teamPrincipal'] = value
                            elif 'world championships' in title: team_data['championships'] = int(re.sub(r'\D', '', value)) if value else 0
                            elif 'podiums' in title: team_data['podiums'] = int(re.sub(r'\D', '', value)) if value else 0
                            elif 'first team entry' in title: team_data['firstTeamEntry'] = int(re.sub(r'\D', '', value)) if value else None
                            elif 'season position' in title: team_data['position'] = int(re.sub(r'\D', '', value)) if value else None
                
                drivers = []
                driver_detail_tags = detail_soup.select("a[data-f1rd-a7s-click='driver_card_click']")
                for driver_link in driver_detail_tags:
                    first_name_tag = driver_link.select_one("p.typography-module_display-l-regular__MOZq8")
                    last_name_tag = driver_link.select_one("p.typography-module_display-l-bold__m1yaJ")
                    
                    full_name = "N/A"
                    if first_name_tag and last_name_tag:
                        full_name = f"{first_name_tag.text.strip()} {last_name_tag.text.strip()}"
                    
                    driver_number = "N/A"
                    driver_page_url = f"{base_url}{driver_link['href']}"
                    try:
                        driver_page_res = requests.get(driver_page_url, headers=headers)
                        driver_page_res.raise_for_status()
                        driver_soup = BeautifulSoup(driver_page_res.content, 'html.parser')
                        
                        p_tags = driver_soup.select("p.typography-module_body-xs-semibold__Fyfwn.typography-module_lg_body-s-compact-semibold__cpAmk")
                        if p_tags:
                            last_p_tag = p_tags[-1]
                            print(last_p_tag)
                            if last_p_tag.text.strip().isdigit():
                                driver_number = last_p_tag.text.strip()

                    except requests.RequestException as e:
                        print(f"Unable to load the driver page {full_name}: {e}")

                    drivers.append({"name": full_name, "number": driver_number})
                
                team_data['drivers'] = drivers
                all_teams_data.append(team_data)

            except requests.RequestException as e:
                print(f"Error while retrieving details for {team_data['name']}: {e}")
                all_teams_data.append(team_data)

        return all_teams_data

    except requests.RequestException as e:
        print(f"Critical error while scraping the teams home page: {e}")
        return None

def scrape_driver_standings_sync(year: int):
    url = f"https://www.formula1.com/en/results/{year}/drivers"
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one('table.f1-table.f1-table-with-data')
        if not table:
            return None
        standings = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 5:
                pos = cols[0].text.strip()
                
                driver_link_tag = cols[1].find('a')
                driver_href = driver_link_tag['href'] if driver_link_tag else ''
                
                driver_id = ''
                driver_slug = ''
                match = re.search(r'/drivers/([A-Z0-9]+)/([^/]+)', driver_href)
                if match:
                    driver_id = match.group(1)
                    driver_slug = match.group(2).replace('.html', '')

                first_name_span = cols[1].select_one('span.max-lg\\:hidden')
                last_name_span = cols[1].select_one('span.max-md\\:hidden')
                
                if first_name_span and last_name_span:
                    driver = f"{first_name_span.text.strip()} {last_name_span.text.strip()}"
                else:
                    driver = cols[1].text.strip()
                
                nationality = cols[2].text.strip()
                car = cols[3].text.strip()
                pts = cols[4].text.strip()
                standings.append({
                    "position": pos, "driver": driver, "nationality": nationality, 
                    "car": car, "points": pts, "driverId": driver_id, "driverSlug": driver_slug
                })
        return standings
    except Exception as e:
        print(f"Error while scraping driver rankings: {e}")
        return None

def scrape_constructor_standings_sync(year: int):
    url = f"https://www.formula1.com/en/results/{year}/team"
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one('table.f1-table.f1-table-with-data')
        if not table:
            return None
        standings = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 3:
                pos = cols[0].text.strip()
                team = cols[1].text.strip()
                pts = cols[2].text.strip()
                standings.append({"position": pos, "team": team, "points": pts})
        return standings
    except Exception as e:
        print(f"Error while scraping the constructors standings: {e}")
        return None

def scrape_driver_results_sync(year: int, driver_id: str, driver_slug: str):
    url = f"https://www.formula1.com/en/results/{year}/drivers/{driver_id}/{driver_slug}"
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one('table.f1-table.f1-table-with-data')
        if not table:
            return None
        results = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if len(cols) >= 5:
                gp_cell = cols[0]
                svg_tag = gp_cell.find('svg')
                if svg_tag:
                    svg_tag.decompose()
                gp = gp_cell.text.strip()

                date = cols[1].text.strip()
                team = cols[2].text.strip()
                race_pos = cols[3].text.strip()
                pts = cols[4].text.strip()
                results.append({
                    "grandPrix": gp, "date": date, "team": team,
                    "racePosition": race_pos, "points": pts
                })
        return results
    except Exception as e:
        print(f"Error while scraping driver results {driver_id}: {e}")
        return None

def get_all_race_results_sync(year: int):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = schedule[schedule['EventDate'].dt.date < datetime.utcnow().date()]
    all_results = []

    for index, race in races.iterrows():
        try:
            session = fastf1.get_session(year, race['RoundNumber'], 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            winner = session.results.iloc[0]
            winner_name = f"{winner['FirstName']} {winner['LastName']}"
            
            fastest_lap = session.laps.pick_fastest()
            fastest_driver = session.results[session.results['DriverNumber'] == fastest_lap['DriverNumber']].iloc[0]
            fastest_driver_name = f"{fastest_driver['FirstName']} {fastest_driver['LastName']}"
            
            fastest_lap_time = fastest_lap['LapTime']
            minutes, seconds = divmod(fastest_lap_time.total_seconds(), 60)
            fastest_lap_time_str = f"{int(minutes):01}:{seconds:06.3f}"

            all_results.append({
                "grandPrix": race['EventName'],
                "winner": winner_name,
                "fastestLapDriver": f"{fastest_driver_name} ({fastest_lap_time_str})",
                "roundNumber": int(race['RoundNumber'])
            })
        except Exception as e:
            print(f"Nie można załadować danych dla {race['EventName']}: {e}")
            continue
    return all_results

def get_specific_race_results_sync(year: int, round_number: int):
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()
        
        results = session.results.copy()
        if results.empty:
            return []
        
        fastest_lap = session.laps.pick_fastest()
        fastest_lap_driver_number = fastest_lap['DriverNumber'] if pd.notna(fastest_lap['DriverNumber']) else None

        results['isFastestLap'] = results['DriverNumber'] == fastest_lap_driver_number

        winner = results.iloc[0]
        winner_laps = winner['Laps']
        winner_time = winner['Time'] if isinstance(winner['Time'], pd.Timedelta) else None

        def format_time_or_status(row):
            position = row['Position']
            laps = row['Laps']
            time = row['Time']
            status = row['Status']

            if position == 1 and isinstance(time, pd.Timedelta):
                total_seconds = time.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds_rem = divmod(remainder, 60)
                return f"{int(hours)}:{int(minutes):02d}:{seconds_rem:06.3f}"

            if status == 'Finished' or status == 'Lapped':
                lap_diff = winner_laps - laps
                if lap_diff > 0:
                    return f"+{int(lap_diff)} lap" if lap_diff == 1 else f"+{int(lap_diff)} laps"
                
                if isinstance(time, pd.Timedelta) and winner_time:
                    total_seconds = time.total_seconds()
                    minutes, seconds = divmod(total_seconds, 60)
                    
                    if minutes >= 1:
                        return f"+{int(minutes):02d}:{seconds:06.3f}"
                    else:
                        return f"+{seconds:.3f}"

            return status

        results['formattedTime'] = results.apply(format_time_or_status, axis=1)
            
        classification = []
        for index, driver in results.iterrows():
            classification.append({
                "position": int(driver['Position']),
                "driverName": f"{driver['FirstName']} {driver['LastName']}",
                "teamName": driver['TeamName'],
                "laps": int(driver['Laps']),
                "timeOrStatus": driver['formattedTime'],
                "points": int(driver['Points']),
                "isFastestLap": bool(driver['isFastestLap'])
            })
        return classification
    except Exception as e:
        print(f"Error while retrieving detailed results for the race {round_number} in year {year}: {e}")
        return None

def scrape_race_results_sync(year: int):
    base_url = "https://www.formula1.com"
    races_url = f"{base_url}/en/results/{year}/races"
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    try:
        response = requests.get(races_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one('table.f1-table.f1-table-with-data')
        if not table:
            return []
        
        all_results = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if len(cols) < 4:
                continue

            gp_cell = cols[0]
            svg_tag = gp_cell.find('svg')
            if svg_tag:
                svg_tag.decompose()
            gp_name = gp_cell.text.strip()
            
            gp_link_tag = cols[0].find('a')
            gp_name = gp_link_tag.text.strip()
            race_link = gp_link_tag['href']
            
            winner_cell = cols[2]
            first_name = winner_cell.select_one('span.max-lg\\:hidden').text.strip()
            last_name = winner_cell.select_one('span.max-md\\:hidden').text.strip()
            winner_name = f"{first_name} {last_name}"

            fastest_lap_link = race_link.replace('race-result', 'fastest-laps')
            fl_response = requests.get(base_url + fastest_lap_link, headers=headers)
            fl_soup = BeautifulSoup(fl_response.content, 'html.parser')
            fl_table = fl_soup.select_one('table.f1-table.f1-table-with-data')
            fastest_lap_driver = "N/A"
            if fl_table:
                first_row = fl_table.find('tbody').find('tr')
                if first_row:
                    fl_cols = first_row.find_all('td')
                    if len(fl_cols) >= 7:
                        fl_driver_cell = fl_cols[2]
                        fl_first_name = fl_driver_cell.select_one('span.max-lg\\:hidden').text.strip()
                        fl_last_name = fl_driver_cell.select_one('span.max-md\\:hidden').text.strip()
                        fl_driver_name = f"{fl_first_name} {fl_last_name}"
                        fl_time = fl_cols[5].text.strip()
                        fastest_lap_driver = f"{fl_driver_name} ({fl_time})"

            all_results.append({
                "grandPrix": gp_name,
                "winner": winner_name,
                "fastestLapDriver": fastest_lap_driver,
                "raceLink": race_link
            })
        return all_results
    except Exception as e:
        print(f"Error while scraping race results for the year {year}: {e}")
        return None


def scrape_specific_race_results_sync(race_link: str):
    base_url = "https://www.formula1.com"
    full_url = base_url + race_link
    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.select_one('table.f1-table.f1-table-with-data')
        if not table:
            return []
            
        classification = []
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if len(cols) < 7:
                continue
            
            driver_cell = cols[2]
            first_name = driver_cell.select_one('span.max-lg\\:hidden').text.strip()
            last_name = driver_cell.select_one('span.max-md\\:hidden').text.strip()
            driver_name = f"{first_name} {last_name}"
            
            team_cell = cols[3]
            team_name = team_cell.text.strip()

            classification.append({
                "position": cols[0].text.strip(),
                "driverName": driver_name,
                "teamName": team_name,
                "laps": cols[4].text.strip(),
                "timeOrStatus": cols[5].text.strip(),
                "points": cols[6].text.strip(),
                "isFastestLap": False
            })
        return classification
    except Exception as e:
        print(f"Error while scraping detailed results from the link {race_link}: {e}")
        return None

@app.get("/api/standings/drivers/{year}")
async def get_driver_standings(year: int):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, scrape_driver_standings_sync, year)
    if data is None:
        raise HTTPException(status_code=500, detail="The driver rankings could not be retrieved.")
    return JSONResponse(content=data)

@app.get("/api/standings/constructors/{year}")
async def get_constructor_standings(year: int):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, scrape_constructor_standings_sync, year)
    if data is None:
        raise HTTPException(status_code=500, detail="The constructor rankings could not be retrieved.")
    return JSONResponse(content=data)

@app.get("/api/results/races/{year}")
async def get_race_results(year: int):
    loop = asyncio.get_event_loop()
    current_year = datetime.now().year
    
    if year == current_year:
        data = await loop.run_in_executor(None, get_all_race_results_sync, year)
    else:
        data = await loop.run_in_executor(None, scrape_race_results_sync, year)

    if data is None:
        raise HTTPException(status_code=500, detail="The race results could not be retrieved.")
    return JSONResponse(content=data)

@app.post("/api/results/race-scraped")
async def get_scraped_race_result(payload: dict):
    race_link = payload.get("raceLink")
    if not race_link:
        raise HTTPException(status_code=400, detail="No “raceLink” in the query.")
    
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, scrape_specific_race_results_sync, race_link)
    
    if data is None:
        raise HTTPException(status_code=500, detail=f"The results for the race could not be retrieved from the link: {race_link}.")
    return JSONResponse(content=data)

@app.get("/api/results/driver/{year}/{driver_id}/{driver_slug}")
async def get_driver_results(year: int, driver_id: str, driver_slug: str):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, scrape_driver_results_sync, year, driver_id, driver_slug)
    if data is None:
        raise HTTPException(status_code=500, detail=f"Unable to retrieve results for driver {driver_id}.")
    return JSONResponse(content=data)

@app.get("/api/results/race/{year}/{round_number}")
async def get_specific_race_result(year: int, round_number: int):
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, get_specific_race_results_sync, year, round_number)
    if data is None:
        raise HTTPException(status_code=500, detail=f"Unable to retrieve results for the race {round_number} in year {year}.")
    return JSONResponse(content=data)


@app.get("/api/teams/{year}")
async def get_teams_data(year: int):
    loop = asyncio.get_event_loop()
    teams_data = await loop.run_in_executor(None, scrape_teams_data_sync, year)
    
    if teams_data is None:
        raise HTTPException(status_code=503, detail="Unable to retrieve team data from the Formula 1 website.")
    
    teams_data.sort(key=lambda x: x.get('position') or 99)
    return JSONResponse(content=teams_data)

@app.get("/api/schedule/{year}")
async def get_schedule(year: int):
    try:
        schedule = await asyncio.to_thread(fastf1.get_event_schedule, year, include_testing=False)
        all_events = []
        for index, event in schedule.iterrows():
            session_dates = {
                'Practice 1': event['Session1DateUtc'], 'Practice 2': event['Session2DateUtc'],
                'Practice 3': event['Session3DateUtc'], 'Qualifying': event['Session4DateUtc'],
                'Race': event['Session5DateUtc'],
            }
            if 'SprintDateUtc' in event and pd.notna(event['SprintDateUtc']):
                session_dates['Sprint'] = event['SprintDateUtc']

            for session_name, session_date in session_dates.items():
                if pd.notna(session_date):
                    all_events.append({
                        "title": f"{session_name} - {event['EventName']}",
                        "date": session_date.isoformat()
                    })
        return all_events
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while downloading schedule: {str(e)}")

@app.get("/api/circuits/{year}")
async def get_circuits(year: int):
    try:
        schedule = await asyncio.to_thread(fastf1.get_event_schedule, year, include_testing=False)
        unique_circuits_df = schedule.drop_duplicates(subset=['Location'])
        circuits_info_df = unique_circuits_df[['Location', 'Country']].copy()
        circuits_info_df.rename(columns={'Location': 'CircuitName'}, inplace=True)
        return circuits_info_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while downloading the track list: {str(e)}")
    
def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    return np.matmul(rot_mat, xy)

@app.get("/api/circuit-details/{year}/{location}")
async def get_circuit_details(year: int, location: str):
    try:
        loop = asyncio.get_event_loop()
        session = None
        lap = None
        
        current_year = datetime.now().year
        map_year = year
        if year >= current_year:
            map_year = year - 1

        for session_type in ['R', 'Q', 'FP3', 'FP2', 'FP1']:
            temp_session = await loop.run_in_executor(None, load_session_data, map_year, location, session_type)
            
            if temp_session is not None:
                temp_lap = temp_session.laps.pick_fastest()
                if temp_lap is not None and not temp_lap.empty:
                    session = temp_session
                    lap = temp_lap
                    break
        
        if session is None or lap is None:
            raise HTTPException(status_code=404, detail=f"No complete data could be found for the track {location} in the season {map_year}.")

        telemetry = lap.get_telemetry().add_distance()
        circuit_info = session.get_circuit_info()

        def assign_sector(dist, circuit_info):
            if not circuit_info or not hasattr(circuit_info, 'sector_locations') or not circuit_info.sector_locations:
                return 1
            s1, s2, s3 = circuit_info.sector_locations.values()
            if dist < s2[0]: return 1
            elif dist < s3[0]: return 2
            else: return 3
        
        telemetry['Sector'] = telemetry['Distance'].apply(assign_sector, args=(circuit_info,))

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('none')
        fig.set_facecolor('none')

        cmap = ListedColormap(['#e10600', '#0090ff', '#f9b000'])
        x = telemetry['X'].to_numpy()
        y = telemetry['Y'].to_numpy()
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        sector_data = telemetry['Sector'].to_numpy(dtype=int) - 1
        lc_sector = LineCollection(segments, cmap=cmap, linewidth=5)
        lc_sector.set_array(sector_data)
        ax.add_collection(lc_sector)

        start_index = telemetry['Distance'].idxmin()
        start_pos = telemetry.loc[start_index, ['X', 'Y']]
        ax.plot(start_pos['X'], start_pos['Y'], 's', markersize=12, color='white', markeredgecolor='black', label='Start/Finish')

        if circuit_info:
            offset_vector = [600, 0]
            for _, corner in circuit_info.corners.iterrows():
                txt = f"{corner['Number']}"
                offset_angle = corner['Angle'] / 180 * np.pi
                offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
                text_x = corner['X'] + offset_x
                text_y = corner['Y'] + offset_y
                ax.text(text_x, text_y, txt, color='white', fontsize=24, ha='center', va='center')

        margin = 1000
        ax.set_xlim(np.min(x) - margin, np.max(x) + margin)
        ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=150)
        buf.seek(0)
        track_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        track_length_km = telemetry['Distance'].max() / 1000
        fastf1_data = {
            'turns': len(circuit_info.corners) if circuit_info else None,
            'length': f"{track_length_km:.3f} km",
            'locationName': session.event['EventName']
        }

        f1_com_data = {}
        try:
            years_to_try = [year]
            if year >= datetime.now().year:
                years_to_try.append(year - 1)
            
            scrape_success = False
            for scrape_year in years_to_try:
                schedule = await asyncio.to_thread(fastf1.get_event_schedule, scrape_year, include_testing=False)
                event_row = schedule[schedule['Location'].str.lower() == location.lower()]
                
                if not event_row.empty:
                    event_name = event_row.iloc[0]['EventName']
                    event_name_lower = event_name.lower()

                    EVENT_SLUG_MAPPING = {
                        'emilia romagna': 'emiliaromagna',
                        'saudi arabian': 'saudi-arabia',
                        'miami': 'miami',
                        'british': 'great-britain',
                        'united states': 'united-states',
                        'las vegas': 'las-vegas',
                        'abu dhabi': 'united-arab-emirates', 
                        'mexico city': 'mexico',
                        'são paulo': 'brazil',
                        'australian': 'australia',
                        'spanish': 'spain',
                        'monaco': 'monaco',
                        'canadian': 'canada',
                        'austrian': 'austria',
                        'hungarian': 'hungary',
                        'belgian': 'belgium',
                        'dutch': 'netherlands',
                        'italian': 'italy',
                        'azerbaijan': 'azerbaijan',
                        'singapore': 'singapore',
                        'japanese': 'japan',
                        'qatar': 'qatar',
                        'chinese': 'china',
                        'bahrain': 'bahrain'
                    }
                    
                    slug = None
                    for key, value in EVENT_SLUG_MAPPING.items():
                        if key in event_name_lower:
                            slug = value
                            break
                    
                    if not slug:
                        continue

                    f1_url = f"https://www.formula1.com/en/racing/{scrape_year}/{slug}"
                    print(f"Attempt to scrape F1.com URL: {f1_url}")

                    headers = {'User-Agent': 'F1Dashboard/1.0 (example-contact@example.com)'}
                    response = await loop.run_in_executor(None, lambda: requests.get(f1_url, headers=headers))

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        info_container = soup.select_one("dl.grid")

                        if info_container:
                            info_items = info_container.find_all("div", recursive=False)

                            for item in info_items:
                                title_tag = item.find("dt")
                                value_tag = item.find("dd")
                                
                                if title_tag and value_tag:
                                    title = title_tag.text.strip().lower()
                                    value = value_tag.text.strip()

                                    if 'number of laps' in title:
                                        f1_com_data['numberOfLaps'] = value
                                    elif 'race distance' in title:
                                        f1_com_data['raceDistance'] = value
                                    elif 'fastest lap' in title or 'lap record' in title:
                                        extra_info_tag = item.find("span")
                                        if extra_info_tag and extra_info_tag.text.strip():
                                            extra_info = extra_info_tag.text.strip()
                                            full_record = f"{value} {extra_info}"
                                            f1_com_data['lapRecord'] = full_record
                                        else:
                                            f1_com_data['lapRecord'] = value

                        scrape_success = True
                        print(f"Success! Scraping completed for {f1_url}")
                        break 
                    else:
                        print(f"Warning: Failed to retrieve data from {f1_url}. Status code: {response.status_code}")
                else:
                    print(f"Warning: No events found for location “{location}” in the schedule for year {scrape_year}.")

            if not scrape_success:
                print("Warning: Scraping failed for all attempts.")

        except Exception as scrape_error:
            print(f"An error occurred while scraping F1.com: {scrape_error}")

        final_data = {
            "trackImage": track_image_base64,
            "length": fastf1_data.get('length'),
            "turns": fastf1_data.get('turns'),
            "locationName": fastf1_data.get('locationName'),
            "lapRecord": f1_com_data.get('lapRecord'),
            "numberOfLaps": f1_com_data.get('numberOfLaps'),
            "raceDistance": f1_com_data.get('raceDistance')
        }

        return JSONResponse(content=final_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error has occurred: {str(e)}")
