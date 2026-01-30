import math

def get_day_of_year(weather_data_row, year=None):
  """ Calculates the day of the year (1-366). Needs error handling. """
  # (Using the robust version from previous answers)
  try:
    month = int(weather_data_row[0])
    day = int(weather_data_row[1])
  except (IndexError, ValueError):
    print("Error: Invalid weather data row format for get_day_of_year.")
    return None
  if not (1 <= month <= 12): return None
  days_in_month_non_leap = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  days_in_month_leap     = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  is_leap = False
  if year is not None:
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0): is_leap = True
  days_list = days_in_month_leap if is_leap else days_in_month_non_leap
  if not (1 <= day <= days_list[month]): return None # Assumes day >= 1
  day_of_year_val = sum(days_list[m] for m in range(1, month)) + day
  return day_of_year_val


def sunPos(LT, phi, lamb, lamb_s, J):
    """
    Returns the Solar Altitide and Solar Azimuth 
    acc. to formulas (D.1) - (D.7) in EN 17037:2018  

    Parameters:
    LT     -- Local Clock Time
    phi    -- geographical latitude of the site
    lamb   -- geographical longitude of the East (+) or West (-) of Greenwich
    lamb_s -- longitude of standard meridian
    J      -- day of year (1 - 365) 
        
    Returns:
    sun_alt -- solar altitude
    sun_azi -- soalr azimuth
    """
    
    # J'
    Jp = 360.0 * J / 365.0
    
    # Equation of time (ET) (in minutes!)
    ET = 0.0066 + \
        7.3525 * math.cos(math.radians(Jp + 85.9)) + \
        9.9359 * math.cos(math.radians(2 * Jp + 108.9)) + \
        0.3387 * math.cos(math.radians(3 * Jp + 105.2))

    # Solar declination (delta)
    delta = 0.3948 - \
        23.2559 * math.cos(math.radians(Jp + 9.1)) - \
        0.3915 * math.cos(math.radians(2 * Jp + 5.4)) - \
        0.1764 * math.cos(math.radians(3 * Jp + 26.0))
        
    # True Solar Time (TST) 
    TST = LT + (lamb - lamb_s)/15.0 + ET/60.0 # ET in minutes!
    
    # Hour angle (omega_eta)
    omega_eta = (12.0 - TST) * 15.0
    
    # convert angles to radiant 
    phi_rad = math.radians(phi)
    omega_eta_rad = math.radians(omega_eta)
    delta_rad = math.radians(delta)
    
    # Solar altitude
    asin_arg = \
        math.cos(omega_eta_rad) * math.cos(phi_rad) * math.cos(delta_rad) + \
        math.sin(phi_rad) * math.sin(delta_rad)
    gamma_s = math.asin(asin_arg)
    sun_alt = math.degrees(gamma_s)

    # Solar Azimuth
    acos_arg = \
        (math.sin(gamma_s) * math.sin(phi_rad) - math.sin(delta_rad)) / \
        (math.cos(gamma_s) * math.cos(phi_rad))
    acos_val_deg = math.degrees(math.acos(acos_arg))

    if TST <= 12.0:
        alpha_s = 180.0 - acos_val_deg
    else:
        alpha_s = 180.0 + acos_val_deg
    sun_azi = alpha_s
    
    return [sun_alt, sun_azi]