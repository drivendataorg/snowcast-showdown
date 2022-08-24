import pandas as pd
import numpy as np

from datetime import date, datetime

def julian_day(year, month, day):
    jd = (1461 * (year + 4800 + (month - 14) / 12 ) ) / 4
    jd += (367 * (month - 2 - 12 * ((month - 14) / 12)) ) / 12
    jd -= (3 * ((year + 4900 + (month - 14) / 12) / 100)) / 4
    jd += day - 32075
    return jd

def julian_century(jd):
    return (jd - 2451545) / 36525

def geo_mean_long_sun(jc):
    """Geom Mean Long Sun (deg)"""
    return (280.46646 + jc *(36000.76983+ jc * 0.0003032)) % 360

def sun_app_long(jc):
    """Sun App Long (deg)"""
    return M2 - 0.00569 - 0.00478*np.sin(np.pi * (125.04 - 1934.136 * jc) / 180)

def geom_mean_anom_sun(jc):
    """Geom Mean Anom Sun (deg)"""
    return 357.52911 + jc*(35999.05029 - 0.0001537 * jc)

def eccent_earth_orbit(jc):
    """Eccent Earth Orbit"""
    return 0.016708634 - jc * (0.000042037 + 0.0000001267*jc)

def sun_eq_of_ctr(gm, jc):
    """Sun Eq of Ctr"""
    gm = np.pi * gm / 180
    r = np.sin(gm)*(1.914602-jc*(0.004817+0.000014*jc))
    r += np.sin(2*gm)*(0.019993-0.000101*jc)+np.sin(3*gm)*0.000289
    return r

def true_sun_long(gm , sun_ctr):
    """Sun True Long (deg)"""
    return gm + sun_ctr

def true_sun_anom_long(ga , sun_ctr):
    """Sun True Anom (deg)"""
    return ga + sun_ctr

def sun_app_long(sun_true_long , jc):
    """Sun App Long (deg)"""
    return sun_true_long - 0.00569 - 0.00478 * np.sin(np.pi*(125.04-1934.136*jc) / 180)

def mean_obliq_ecliptic(jc):
    """Mean Obliq Ecliptic (deg)"""
    return 23+(26+((21.448-jc*(46.815+jc*(0.00059 - jc*0.001813))))/60)/60

def obliq_corr(jc):
    """Obliq Corr (deg)"""
    mean_obliq = mean_obliq_ecliptic(jc)
    return mean_obliq + 0.00256 * np.cos(np.pi * (125.04-1934.136*jc) / 180)

def sun_app_long_full(jc):

    gm = geo_mean_long_sun(jc)
    gam = geom_mean_anom_sun(jc)
    sun_ctr = sun_eq_of_ctr(gam, jc)
    tsl = true_sun_long(gm , sun_ctr)
    return sun_app_long(tsl , jc)

def sun_declin(jc):
    """
        Sun Declin (deg)
    oc - Obliq Corr (deg)
    sa - Sun App Long (deg)
    """
    oc = np.pi * obliq_corr(jc) / 180
    sa = np.pi * sun_app_long_full(jc) / 180

    r = np.arcsin( np.sin(oc) * np.sin(sa) )
    return r * 180 / np.pi

def sun_declin_rad(jc):
    """
        Sun Declin (rad)
    oc - Obliq Corr (deg)
    sa - Sun App Long (deg)
    """
    oc = np.pi * obliq_corr(jc) / 180
    sa = np.pi * sun_app_long_full(jc) / 180

    r = np.arcsin( np.sin(oc) * np.sin(sa) )
    return r

def ha_sunrise(lat, year, month, day):
    """HA Sunrise (deg)"""
    lat_rad = np.pi * lat / 180
    jc = julian_century(
            julian_day(year, month, day))
    decl_rad = sun_declin_rad(jc)
    p1  = np.cos(np.pi * 90.833 / 180) / (np.cos(lat_rad) * np.cos(decl_rad))
    p2 = np.tan(lat_rad) * np.tan(decl_rad)
    return np.arccos(p1 - p2)

sun_decline = pd.DataFrame(columns=['decline', 'decline_rad', 'tan_decl', 'cos-1_decl'],
                           index=pd.date_range("2010-01-01", "2025-01-01", freq="1D"))
sun_decline.index.name = 'timestamp'

y, m, d = sun_decline.index.year, sun_decline.index.month, sun_decline.index.day

sun_decline['decline'] = sun_declin(
    julian_century(julian_day(y, m, d)))

sun_decline['decline_rad'] = np.pi * sun_decline['decline'] / 180
sun_decline['tan_decl'] = np.tan(sun_decline['decline_rad'])
sun_decline['cos-1_decl'] = 1 / np.cos(sun_decline['decline_rad'])

sun_decline.to_csv("sun_decline.csv")
