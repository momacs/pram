disease_name = 'COVID-19'

# Prob. of fatality (https://www.worldometers.info/coronavirus/coronavirus-age-sex-demographics):
p_covid19_fat_by_age_group = {
    '0-9'   : 0.000,
    '10-19' : 0.002,
    '20-29' : 0.002,
    '30-39' : 0.002,
    '40-49' : 0.004,
    '50-59' : 0.013,
    '60-69' : 0.036,
    '70-79' : 0.080,
    '80+'   : 0.148
}
p_covid19_fat_by_age_group_comb = {  # similar age groups combined for performance
    '0-50'  : 0.002,
    '50-59' : 0.013,
    '60-69' : 0.036,
    '70-79' : 0.080,
    '80+'   : 0.148
}
p_covid19_fat = sum(p_covid19_fat_by_age_group.values()) / len(p_covid19_fat_by_age_group)


# Prob. of hypertension (https://www.americashealthrankings.org/explore/annual/measure/Hypertension/state/PA):
p_ht_by_age_group = {
    '0-9'   : 0.000,
    '10-19' : 0.000,
    '20-29' : 0.137,
    '30-39' : 0.137,
    '40-49' : 0.259,  # average of the two adjacent data points
    '50-59' : 0.382,
    '60-69' : 0.493,  # average of the two adjacent data points
    '70-79' : 0.605,
    '80+'   : 0.605
}
p_ht = sum(p_ht_by_age_group.values()) / len(p_ht_by_age_group)


# Prob. of fatality due to cardiovascular disease (CVD) implicating hypertention (https://www.cdc.gov/nchs/data/nvsr/nvsr68/nvsr68_09-508.pdf; Table 7, pg 38):
p_cvd_fat_by_age_group = {  #
    '0-9'   : 0.0,  # excluded due to focus on risks specific to adult population (and negligeable in kids anyway)
    '10-19' : 0.0,  # ^
    '20-29' : ((  2.6 +   10.1) / 2) / 100_000,
    '30-39' : (( 10.1 +   32.2) / 2) / 100_000,
    '40-49' : (( 32.2 +   95.9) / 2) / 100_000,
    '50-59' : (( 95.9 +  237.2) / 2) / 100_000,
    '60-69' : ((237.2 +  505.6) / 2) / 100_000,
    '70-79' : ((505.6 + 1391.3) / 2) / 100_000,
    '80+'   : 5249.9                 / 100_000,
}
p_cvd_fat = 262.3 / 100_000  # https://www.cdc.gov/nchs/data/nvsr/nvsr68/nvsr68_09-508.pdf; Table 7, pg 38
