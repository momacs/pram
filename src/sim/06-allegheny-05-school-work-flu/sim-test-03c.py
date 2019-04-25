'''
The simple flu model on the Allegheny County data (students only).

Identify good school to investigate:

SELECT COUNT(*) AS n, st.school_id, AVG(st.income) AS income_m, st.income_level, AVG(age) AS age_m, sc.latitude, sc.longitude
FROM students st INNER JOIN schools sc on sc.sp_id = st.school_id GROUP BY sc.sp_id, st.income_level ORDER BY n DESC, sc.sp_id

n     school_id   income_mean         i   age_mean            lat lon
----------------------------------------------------------------------------------
6386    450066968    134790.52709051    m    5.70012527403696    0.0    0.0                     m01
5655    450086847    129025.519186561    m    13.8258178603006    40.632399    -80.4466
3294    450086847    26074.3989071038    l    13.3023679417122    40.632399    -80.4466
2928    450066968    28083.0925546448    l    5.94330601092896    0.0    0.0                 m01
2587    450067304    145389.131812911    m    12.0552763819095    0.0    0.0
1987    450063600    163197.256165073    m    15.9869149471565    0.0    0.0                 m02
1792    450109054    144399.965959821    m    16.0848214285714    40.33306    -80.043229
1471    450143076    141918.295717199    m    16.1067301155676    40.376388    -80.050603
1355    450107620    124959.326937269    m    16.0236162361624    40.355898    -79.979719   m03
1255    450127687    136100.890836653    m    16.1003984063745    40.58123    -79.942932
1251    450067304    26346.5659472422    l    12.2134292565947    0.0    0.0
1125    450144545    147972.320888889    m    16.1066666666667    40.571955    -80.031351
1124    450114199    139027.40569395    m    16.1085409252669    40.378624    -80.110997
1108    450102516    116780.759927798    m    15.4837545126354    40.417125    -80.020406
1078    450125133    111076.614100186    m    16.0556586270872    40.429923    -79.751352
1018    450102513    25104.4184675835    l    16.0284872298625    40.429871    -79.920355
963    450140462    27159.015576324    l    16.202492211838    40.343793    -79.829742
954    450149633    108054.22851153    m    16.0649895178197    40.508254    -79.773995
947    450142450    108652.442449842    m    16.1045406546991    40.512505    -80.215362
936    450102513    105304.568376068    m    15.9957264957265    40.429871    -79.920355
916    450148402    102050.793668122    m    16.1495633187773    40.468266    -79.81889
899    450148402    23912.5806451613    l    15.9788654060067    40.468266    -79.81889
880    450140462    87066.2022727273    m    15.9681818181818    40.343793    -79.829742
864    450120864    148946.793981481    m    10.5636574074074    40.65769    -79.96114
830    450144841    124331.396385542    m    14.0518072289157    40.52379    -80.02957
805    450123912    134385.561490683    m    16.2086956521739    40.517303    -79.869527
785    450142414    111161.197452229    m    16.0509554140127    40.467685    -80.115075
759    450063600    25748.2608695652    l    15.6916996047431    0.0    0.0                      m02
755    450144546    166684.495364238    m    16.0304635761589    40.605004    -80.05099
751    450102525    139161.288948069    m    16.0652463382157    40.488982    -80.017937
728    450149292    18925.1236263736    l    8.16483516483516    40.452781    -79.893929       l01
721    450142449    113066.205270458    m    12.1983356449376    40.5097    -80.222883
711    450144842    117614.555555556    m    16.2728551336146    40.523738    -80.0296
705    450137243    24313.5007092199    l    12.4524822695035    40.482632    -79.814739
695    450137243    92273.2805755396    m    12.0187050359712    40.482632    -79.814739
684    450067740    217273.074561404    m    8.26754385964912    40.6262522    -80.0561964
684    450110131    166532.228070175    m    12.1520467836257    40.315095    -80.098836
682    450102517    126804.721407625    m    8.69648093841642    40.389005    -80.007637
682    450143658    141804.609970675    m    12.2140762463343    40.305118    -80.022754
674    450127690    148147.166172107    m    12.1038575667656    40.589777    -79.945388
669    450114201    142415.784753363    m    12.2167414050822    40.378624    -80.110997
669    450123912    24042.466367713    l    16.0822122571001    40.517303    -79.869527
656    450140545    150377.222560976    m    8.41920731707317    40.572011    -80.028754
650    450102516    26042.4707692308    l    15.1338461538462    40.417125    -80.020406
638    450139597    174223.855799373    m    8.29937304075235    40.641729    -80.093503
622    450149286    23422.6173633441    l    14.6768488745981    40.444962    -79.998882
616    450102518    132250.548701299    m    8.79707792207792    40.451087    -80.049011
616    450134114    126814.237012987    m    16.0714285714286    40.388943    -80.034458
615    450107620    27387.0894308943    l    15.9040650406504    40.355898    -79.979719     m03
614    450119183    134104.633550489    m    12.1530944625407    40.56468    -79.903316
613    450149076    174293.849918434    m    11.9951060358891    40.655193    -80.010701
607    450128119    129650.810543657    m    12.2701812191104    40.3514    -80.002797
604    450131007    140199.850993377    m    12.2119205298013    40.340402    -80.041998
599    450143076    28855.0851419032    l    15.8347245409015    40.376388    -80.050603
595    450139639    180901.166386555    m    12.1361344537815    40.64174    -80.093871
593    450114202    129885.615514334    m    7.20910623946037    40.379123    -80.110204
588    450131157    160928.741496599    m    12.0952380952381    40.57834    -80.072185
585    450149633    25742.8547008547    l    15.9059829059829    40.508254    -79.773995
583    450124148    163431.46483705    m    8.42881646655232    40.586498    -80.094958
574    450121426    98662.668989547    m    15.9337979094077    40.256085    -79.854512
570    450107374    142676.285964912    m    8.38245614035088    40.539606    -80.094764
569    450125133    26461.1458699473    l    15.97539543058    40.429923    -79.751352
563    450150652    147882.95026643    m    16.0248667850799    40.561812    -80.203397
562    450129446    109572.960854093    m    16.085409252669    40.618188    -79.728338
553    450140459    115537.844484629    m    8.62025316455696    40.413465    -80.196843
544    450123763    156298.632352941    m    12.2132352941176    40.354663    -80.06425
542    450049316    216992.41697417    m    10.9428044280443    40.597899    -79.9675864
541    450146354    133122.3974122    m    8.6913123844732    40.516959    -79.863316
539    450112699    138908.515769944    m    12.0519480519481    40.569444    -80.035415
539    450149299    20029.6474953618    l    8.00185528756957    40.460504    -79.912605     l03
535    450149313    24830.3925233645    l    7.1981308411215    40.398001    -79.989084
534    450149287    27158.9887640449    l    15.9007490636704    40.391203    -79.988508
532    450131156    159046.781954887    m    8.38721804511278    40.581957    -80.052425
527    450149306    22905.2941176471    l    7.39848197343454    40.423232    -79.923386
523    450110212    182098.929254302    m    8.565965583174    40.638838    -80.081474
523    450149298    17862.5047801147    l    8.12045889101339    40.454862    -80.005402
519    450149285    99833.2986512524    m    7.55491329479769    40.394882    -80.024949
512    450109857    124343.09375    m    8.462890625    40.546359    -80.239929
493    450107553    152180.831643002    m    7.6369168356998    40.337294    -80.095692
493    450149305    24718.7423935091    l    13.501014198783    40.448801    -79.961581
492    450149310    24300.6178861789    l    14.2337398373984    40.457347    -79.919567
490    450140452    131497.716326531    m    9.23265306122449    40.550046    -80.004724
489    450149287    93150.0572597137    m    16.1676891615542    40.391203    -79.988508
483    450112506    99939.867494824    m    14.8364389233954    40.428239    -80.095631
479    450120925    121707.649269311    m    7.4133611691023    40.503202    -80.07687
477    450066931    220946.660377358    m    12.1635220125786    40.5217565    -79.8794891
477    450075546    164202.264150943    m    10.7672955974843    40.4514804    -79.9423784
474    450113419    157798.075949367    m    8.77004219409283    40.571708    -79.927196
474    450129446    27385.358649789    l    15.8248945147679    40.618188    -79.728338
473    450144842    28510.6596194503    l    16.1035940803383    40.523738    -80.0296
470    450121197    153004.212765957    m    7.65744680851064    40.348236    -80.063222
467    450107375    151091.299785867    m    16.0107066381156    40.529878    -80.075854
465    450102519    23726.6279569892    l    6.25806451612903    40.46877    -79.917847
461    450148400    84742.7462039045    m    7.61388286334056    40.45898    -79.826429
457    450149284    21898.9059080963    l    6.96498905908096    40.415647    -80.02084
455    450078733    112539.265934066    m    16.0813186813187    40.44265    -79.999317
455    450120959    164525.072527473    m    8.81538461538462    40.547807    -80.190828
453    450118168    133320.710816777    m    16.0485651214128    40.618295    -79.852952
452    450068129    112742.681415929    m    8.32964601769912    40.4412821    -79.771786
451    450102519    109135.521064302    m    6.74944567627494    40.46877    -79.917847
'''

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import FluLocationRule, FluLocationAlleghenyRule, ProgressFluSimpleRule, ProgressFluSimpleAlleghenyRule
from pram.sim    import Simulation


import signal
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# ----------------------------------------------------------------------------------------------------------------------
# (0) Init:

rand_seed = 1928

pragma_live_info = True
pragma_live_info_ts = False

fpath_db_in  = os.path.join(os.path.dirname(__file__), 'allegheny-students.sqlite3')


# ----------------------------------------------------------------------------------------------------------------------
# (1) Sites:

sites = Simulation.gen_sites_from_db(
    fpath_db_in,
    lambda fpath_db: {
        'school': Site.gen_from_db(fpath_db, 'schools', 'sp_id', 'school', [])
    },
    pragma_live_info=pragma_live_info,
    pragma_live_info_ts=pragma_live_info_ts
)

site_home = Site('home')

# ----------------------------------------------------------------------------------------------------------------------
# (2) Probes:

# n_schools = 8
# few_schools = [sites['school'][k] for k in list(sites['school'].keys())[:n_schools]]
#
# probe_grp_size_few_schools = GroupSizeProbe('school', [GroupQry(rel={ Site.AT: s }) for s in few_schools], msg_mode=ProbeMsgMode.DISP)

fpath_db = os.path.join(os.path.dirname(__file__), 'out-test-03c.sqlite3')

if os.path.isfile(fpath_db):
    os.remove(fpath_db)

pp = ProbePersistanceDB(fpath_db)

school_l01 = sites['school'][450149292]
school_l02 = sites['school'][450149286]  # might be worse than l01
school_l03 = sites['school'][450149299]  # ^

school_m01 = sites['school'][450066968]
school_m02 = sites['school'][450063600]
school_m03 = sites['school'][450107620]


def probe_grp_size_flu_school(name, school, pp):
    return GroupSizeProbe(
        name=name,
        queries=[
            GroupQry(attr={ 'flu': 's' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'e' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school })
        ],
        qry_tot=GroupQry(rel={ 'school': school }),
        persistance=pp,
        var_names=['ps', 'pe', 'pr', 'ns', 'ne', 'nr']
    )

probe_grp_size_flu_school_l01 = probe_grp_size_flu_school('school-l01', school_l01, pp)
probe_grp_size_flu_school_l02 = probe_grp_size_flu_school('school-l02', school_l02, pp)
probe_grp_size_flu_school_l03 = probe_grp_size_flu_school('school-l03', school_l03, pp)

probe_grp_size_flu_school_m01 = probe_grp_size_flu_school('school-m01', school_m01, pp)
probe_grp_size_flu_school_m02 = probe_grp_size_flu_school('school-m02', school_m02, pp)
probe_grp_size_flu_school_m03 = probe_grp_size_flu_school('school-m03', school_m03, pp)


# ----------------------------------------------------------------------------------------------------------------------
# (3) Simulation:

(Simulation().
    set().
        rand_seed(rand_seed).
        pragma_autocompact(True).
        pragma_live_info(pragma_live_info).
        pragma_live_info_ts(pragma_live_info_ts).
        pragma_rule_analysis_for_db_gen(True).
        done().
    add().
        # rule(ProgressFluSimpleRule()).
        # rule(FluLocationRule()).
        rule(ProgressFluSimpleAlleghenyRule()).
        rule(FluLocationAlleghenyRule()).
        # probe(probe_grp_size_few_schools).
        probe(probe_grp_size_flu_school_l01).
        probe(probe_grp_size_flu_school_l02).
        probe(probe_grp_size_flu_school_l03).
        probe(probe_grp_size_flu_school_m01).
        probe(probe_grp_size_flu_school_m02).
        probe(probe_grp_size_flu_school_m03).
        done().
    gen_groups_from_db(
        fpath_db_in,
        tbl='students',
        attr={ 'flu': 's' },
        rel={ 'home': site_home },
        attr_db=[],
        rel_db=[
            GroupDBRelSpec('school', 'school_id', sites['school'])
        ],
        rel_at='school'
    ).
    run(3).
    summary(True, 0,0,0,0, (1,0))
)