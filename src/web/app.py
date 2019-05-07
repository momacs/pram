#
# Next
#     Add 'human-name' property to a rule which is displayed in the UI
#     Add Download simulation button
#     Session management
#         Allow saving, loading, deleting, and duplicating sessions (ideally simple via serialized objects)
#             [+] Pickle the Simulation object
#             [ ] Pickle the Flask Session object
#     WebSockets
#         https://blog.miguelgrinberg.com/post/easy-websockets-with-flask-and-gevent
#         https://www.shanelynn.ie/asynchronous-updates-to-a-webpage-with-flask-and-socket-io/
#     REST API
#         https://stackoverflow.com/questions/44430906/flask-api-typeerror-object-of-type-response-is-not-json-serializable
#
# Res
#     Flask and Celery
#         https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
#         http://flask.pocoo.org/docs/1.0/deploying/
#         https://blog.miguelgrinberg.com/post/using-celery-with-flask
#         https://hackersandslackers.com/the-art-of-building-flask-routes/
#         https://github.com/pallets/flask/blob/1.0.2/examples/javascript/js_example/templates/fetch.html
#     Flask deployment
#         https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uwsgi-and-nginx-on-ubuntu-14-04
#     Materialize
#         https://materializecss.com/color.html
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Dev
#     Setup
#         cd /Volumes/d/pitt/sci/pram/
#         bin/activate
#         python -m pip install Flask celery psutil
#         echo 'Install Redis'
#     Redis
#         cd /Volumes/d/pitt/sci/pram/logic/redis-5.0.4/src
#         ./redis-server
#     Celery
#         cd /Volumes/d/pitt/sci/pram/src/web
#         celery worker -A app.celery --loglevel=info
#     Celery - Flower
#         cd /Volumes/d/pitt/sci/pram/src/web
#         flower -A app.celery --port=5555
#     Flask
#         cd /Volumes/d/pitt/sci/pram/src/web
#         FLASK_ENV=development FLASK_APP=web flask run --host=0.0.0.0
#     Materialize
#         brew install sass/sass/sass
#         cd /Volumes/d/pitt/sci/pram/src/web/static
#         sass sass-pram/materialize.scss css/materialize-pram.css
#
# Prod (FreeBSD 12R)
#     Setup
#         sudo pkg install py36-Flask py36-celery redis py36-psutil
#     Redis
#         redis-server
#     Celery
#         cd ~/prj/pram/web
#         celery worker -A app.celery --loglevel=info
#     Celery - Flower
#         cd ~/prj/pram/web
#         flower -A app.celery --port=5555
#     Flask
#         cd ~/prj/pram/web
#         export LANG=en_US.UTF-8
#         export LC_ALL=en_US.UTF-8
#         FLASK_ENV=production FLASK_APP=web /usr/local/bin/flask-3.6 run --host=192.168.0.164 --port=5050
#
#     http://thesnaken.asuscomm.com:5050
#
# ----------------------------------------------------------------------------------------------------------------------

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # pram pkg path


import inspect
import os
import psutil
import shutil

import config

from celery import Celery, states
from celery.task.control import revoke
from collections import OrderedDict
from flask import Flask, current_app, jsonify, request, render_template, Response, session, url_for
from flask_session import Session
# from flask.ext.session import Session

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, GoToAndBackTimeAtRule, ResetSchoolDayRule, SEIRFluRule, TimeAlways, TimePoint
from pram.rule   import SimpleFluLocationRule, SimpleFluProgressRule
from pram.sim    import Simulation
from pram.util   import DB, Size


SUDO_CODE = 'catch22'  # TODO: Use env variable
LOAD_CPU_INT = 1  # CPU load sample interval [s]

PATH_DB = os.path.join(os.path.dirname(__file__), 'db')

DB_FEXT = 'sqlite3'  # database file extension

tasks = {}

# from kombu import Exchange, Queue
# task_queues = (
#     Queue('celery', routing_key='celery'),
#     Queue('transient', Exchange('transient', delivery_mode=1), routing_key='transient', durable=False)
# )


# ----------------------------------------------------------------------------------------------------------------------
# ----[ FLASK ]---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def make_flask():
    app = Flask(__name__)
    app.config.from_object('config.DevConfig')
    Session().init_app(app)
    return app

app = make_flask()


# ----------------------------------------------------------------------------------------------------------------------
# ----[ CELERY ]--------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)
app.app_context().push()

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/status/<task_id>', methods=['GET', 'POST'])
def task_status(task_id):
    # task = sim_02_run_bg.AsyncResult(task_id)
    task = tasks.get(task_id, None)
    if not task:
        sim_clear(task_id)
        return jsonify({ 'res': False, 'i': 0, 'n': 1, 'p': 0, 'isRunning': False, 'isDone': False })

    if task.state == states.PENDING:
        res = { 'res': True, 'i': 0, 'n': 1, 'p': 0, 'isRunning': True, 'isDone': False }
    elif task.state != states.FAILURE:
        res = { 'res': True, 'i': task.info.get('i',0), 'n': task.info.get('n',0), 'p': task.info.get('p',0), 'isRunning': True, 'isDone': (task.state == 'SUCCESS') }
    else:
        res = { 'res': False, 'i': 0, 'n': 1, 'p': 0, 'isRunning': False, 'isDone': True, 'err': str(task.info) }

    if task and task.state in [states.SUCCESS, states.FAILURE, states.REVOKED]:
        sim_clear(task_id)

    return jsonify(res)


# ----------------------------------------------------------------------------------------------------------------------
# ----[ UTIL ]----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ...


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SYS & USER ]----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def session_init(session):
    sim_flu_init(session)
    sim_flu_ac_init(session)

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    session_init(session)
    return render_template('base.html')

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sys-get-load', methods=['GET'])
def sys_get_load():
    ''' Return server load (i.e., resource utilization). '''

    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient access rights' })

    avg_1, avg_5, avg_15 = os.getloadavg()
    cpu_used = psutil.cpu_percent(interval=LOAD_CPU_INT)
    hdd_size, hdd_used, hdd_free = shutil.disk_usage(os.sep)

    return jsonify({
        'res': True,
        'avg': {
            't1'  : f'{round(avg_1,2)}',
            't5'  : f'{round(avg_5,2)}',
            't15' : f'{round(avg_15,2)}'
        },
        'size': {
            'cpu': f'{psutil.cpu_count()} threads',
            'ram': f'{Size.bytes2human(psutil.virtual_memory().total)}',
            'hdd': f'{Size.bytes2human(hdd_size)}'
        },
        'used': {
            'cpu': f'{cpu_used}%',
            'ram': f'{Size.bytes2human(psutil.virtual_memory().used)}',
            'hdd': f'{Size.bytes2human(hdd_used)} ({round(float(hdd_used) / float(hdd_size) * 100, 0)}%)'
        },
        'free': {
            'cpu': f'{100 - cpu_used}%',
            'ram': f'{Size.bytes2human(psutil.virtual_memory().available)}',
            'hdd': f'{Size.bytes2human(hdd_free)} ({round(float(hdd_free) / float(hdd_size) * 100, 0)}%)'
        }
    })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sys-ping', methods=['HEAD'])
def sys_ping():
    res = Response()
    res.headers.add('content-length', 1)
    return res

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-reset-sess', methods=['GET'])
def usr_reset_sess():
    session.clear()
    return usr_get_sess()

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-get-sess', methods=['GET'])
def usr_get_sess():
    if not 'sim-flu-ac' in session:
        sim_flu_ac_init(session)

    return jsonify({ 'res': True, 'state': { 'simFluAC': session['sim-flu-ac'].get_state() }, 'sim': session.get('sim', None) })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-is-root', methods=['GET'])
def usr_is_root():
    ''' Check if the user has elevated access right. '''

    return jsonify({ 'res': True, 'isRoot': session.get('is-root', False) })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-toggle', methods=['POST'])
def usr_toggle():
    ''' Grant the user elevated access rights if the code submitted is correct. '''

    code = request.values.get('code', '', type=str)
    session['is-root'] = (code == SUDO_CODE)
    return jsonify({ 'res': True, 'isRoot': session.get('is-root', False) })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ RULES ]---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class FluProgressRule2(Rule):
    def __init__(self):
        super().__init__('flu-progress', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Susceptible:
        if group.has_attr({ 'flu': 's' }):
            at  = group.get_rel(Site.AT)
            n   = at.get_pop_size()                               # total   population at current location
            n_e = at.get_pop_size(GroupQry(attr={ 'flu': 'e' }))  # exposed population at current location

            p_infection = float(n_e) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

            return [
                GroupSplitSpec(p=    p_infection, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=1 - p_infection, attr_set={ 'flu': 's' })
            ]

        # Exposed:
        if group.has_attr({ 'flu': 'e' }):
            return [
                GroupSplitSpec(p=0.2, attr_set={ 'flu': 'r' }),  # group size after: 20% of before (recovered)
                GroupSplitSpec(p=0.8, attr_set={ 'flu': 'e' })   # group size after: 80% of before (still exposed)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.9, attr_set={ 'flu': 'r' }),
                GroupSplitSpec(p=0.1, attr_set={ 'flu': 's' })
            ]

    def setup(self, pop, group):
        return [
            GroupSplitSpec(p=0.9, attr_set={ 'flu': 's' }),
            GroupSplitSpec(p=0.1, attr_set={ 'flu': 'e' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class FluLocationRule2(Rule):
    def __init__(self):
        super().__init__('flu-location', TimeAlways())

    def apply(self, pop, group, iter, t):
        # Exposed and low income:
        if group.has_attr({ 'flu': 'e', 'income': 'l' }):
            return [
                GroupSplitSpec(p=0.1, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.9)
            ]

        # Exposed and medium income:
        if group.has_attr({ 'flu': 'e', 'income': 'm' }):
            return [
                GroupSplitSpec(p=0.6, rel_set={ Site.AT: group.get_rel('home') }),
                GroupSplitSpec(p=0.4)
            ]

        # Recovered:
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.8, rel_set={ Site.AT: group.get_rel('school') }),
                GroupSplitSpec(p=0.2)
            ]

        return None


# ----------------------------------------------------------------------------------------------------------------------
def rule_get_inf(rule):
    cls = rule if inspect.isclass(rule) else rule.__class__
    return {
        'cls': cls.__name__,
        'name': cls.NAME,
        'docstr': inspect.cleandoc(cls.__doc__).split('\n'),
        'srcLines': inspect.getsourcelines(cls)[0]
    }

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/rules-ls', methods=['GET'])
def rules_ls():
    rules = [
        rule_get_inf(SimpleFluProgressRule),
        rule_get_inf(SimpleFluLocationRule),
        rule_get_inf(SEIRFluRule)
    ]
    return jsonify({ 'res': True, 'rules': rules })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ PROBES ]--------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/probes-ls', methods=['GET'])
def probes_ls():
    school_l  = Site(450149323)  # 88% low income students
    school_m  = Site(450067740)  #  7% low income students

    return GroupSizeProbe(
        name=name or str(school.name),
        queries=[
            GroupQry(attr={ 'flu': 's' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'e' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school })
        ],
        qry_tot=GroupQry(rel={ 'school': school }),
        persistance=pp,
        var_names=['ps', 'pe', 'pr', 'ns', 'ne', 'nr']
    )

    probes = [
        { 'name': 'Low-income school',    'inf': '88% low income students', 'persistenceName': 'low-income', 'id': 450149323 },
        { 'name': 'medium-income school', 'inf':  '7% low income students', 'persistenceName': 'med-income', 'id': 450067740 }
    ]
    return jsonify({ 'res': True, 'probes': probes })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ DB ]------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def db_get_fpath(fname):
    '''
    For security, ensures the filename doesn't contain the 'os.sep' character. Expects filename without the extension
    which it adds. It also does file existance check.
    '''

    if os.sep in fname:
        return None

    fpath = os.path.join(PATH_DB, f'{fname}.{DB_FEXT}')
    if not os.path.isfile(fpath):
        return None

    return fpath

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/db-get-schema', methods=['POST'])
def db_get_schema():
    '''
    Returns the schema of the DB specified by the 'fname' argument.  The tables are sorted by name and the columns are
    kept in the order they appear in the DB.
    '''

    fname = request.values.get('fname', '', type=str)
    fpath = db_get_fpath(fname)
    if not fpath:
        return jsonify({ 'res': False, 'err': 'Incorrect database specified' })

    with DB.open_conn(fpath) as c:
        schema = []
        for r01 in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC").fetchall():  # sql
            tbl = r01['name']
            cols = OrderedDict((r['name'], { 'name': r['name'], 'type': r['type'] }) for r in c.execute(f"PRAGMA table_info('{tbl}')").fetchall())  # cid,name,type,notnull,dflt_value,pk
                # We need a dict here to use it below when iterating through the foreign keys; we eventually convert it
                # into an array though so that the order is preserved on the client side (otherwise, it is lost).

            for row in c.execute(f'PRAGMA foreign_key_list({tbl})').fetchall():  # id,seq,tbl,from,to,on_update,on_delete,match
                cols[row['from']]['fk'] = { 'tbl': row['table'], 'col': row['to'] }

            row_cnt = c.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]

            schema.append({ 'name': tbl, 'cols': [v for v in cols.values()], 'rowCnt': row_cnt })

    return jsonify({ 'res': True, 'schema': schema })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/db-ls', methods=['GET'])
def db_ls():
    '''
    Finds only *.DB_FEXT files.  It subsequently strips the extension from the returned file list and returns only
    filenames without the path.
    '''

    return jsonify({ 'res': True, 'dbs': [f[:-len(f'.{DB_FEXT}')] for f in os.listdir(PATH_DB) if f.endswith(f'.{DB_FEXT}')]})


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SIM ]-----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def sim_clear(task_id):
    if not task_id:
        return

    if tasks.get(task_id, None):
        # revoke(task_id, terminate=True)  # this causes Celery to hang
        tasks.pop(task_id)

    session.pop('sim', None)
    session.pop('task-id', None)

# ----------------------------------------------------------------------------------------------------------------------
def sim_get_pragma_request_value(name, request):
    '''
    The type of the Simulation object's pragma value varies.  This function takes the 'name' of the pragma and the
    'request' object and returns the associated value passed from the client with the appropriate type.

    Unfortunately, Flask does not seem to handle 'type=bool' well so a string comparison is necessary.  This function
    declares a value as True if the associated string is equal to either 'true' or '1'; otherwise, a False is returned.
    '''

    return {
        'analyze'                  : request.values.get('value', '', type=str) in ['true', '1'],
        'autocompact'              : request.values.get('value', '', type=str) in ['true', '1'],
        'autoprune_groups'         : request.values.get('value', '', type=str) in ['true', '1'],
        'autostop'                 : request.values.get('value', '', type=str) in ['true', '1'],
        'autostop_n'               : request.values.get('value', '', type=int),
        'autostop_p'               : request.values.get('value', '', type=float),
        'autostop_t'               : request.values.get('value', '', type=int),
        'live_info'                : request.values.get('value', '', type=str) in ['true', '1'],
        'live_info_ts'             : request.values.get('value', '', type=str) in ['true', '1'],
        'probe_capture_init'       : request.values.get('value', '', type=str) in ['true', '1'],
        'rule_analysis_for_db_gen' : request.values.get('value', '', type=str) in ['true', '1'],
    }.get(name, None)


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SIM: FLU ]------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def sim_flu_init(session, do_force=False):
    if 'sim-flu' in session and not do_force:
        return

    if 'sim-flu' in session:
        session.pop('sim-flu')

    sites = { s:Site(s) for s in ['home', 'school-a', 'school-b']}
    probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.CUMUL)

    session['sim-flu'] = (
        Simulation().
            set().
                pragma_live_info(False).
                pragma_live_info_ts(False).
                done().
            add().
                rule(ResetSchoolDayRule(TimePoint(7))).
                rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
                probe(probe_grp_size_site).
                done().
            new_group(500).
                set_rel(Site.AT,  sites['home']).
                set_rel('home',   sites['home']).
                set_rel('school', sites['school-a']).
                done().
            new_group(500).
                set_rel(Site.AT,  sites['home']).
                set_rel('home',   sites['home']).
                set_rel('school', sites['school-b']).
                done()
    )

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-reset', methods=['GET'])
def sim_flu_reset():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    sim_flu_init(session, True)

    return jsonify({ 'res': True })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-run', methods=['POST'])
def sim_flu_run():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    iter_n = request.values.get('iter-n',  '', type=int)
    sim = session.get('sim-flu', None)
    probe = sim.probes[0]
    probe.clear()
    sim.run(iter_n)

    return jsonify({ 'res': True, 'out': probe.get_msg() })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SIM: FLU ALLEGHENY ]--------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/sim-flu-ac-add-rule', methods=['POST'])
def sim_flu_ac_add_rule():
    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    sim = session['sim-flu-ac']
    cls = request.values.get('cls', '', type=str)
    rule_cls = [r.__class__ for r in sim.rules]

    if cls == 'SimpleFluProgressRule':
        if SimpleFluProgressRule in rule_cls:
            return jsonify({ 'res': False, 'err': 'Rule already in the simulation' })
        sim.add_rule(SimpleFluProgressRule())
        return jsonify({ 'res': True, 'rules': session['sim-flu-ac'].get_state()['rules'] })

    if cls == 'SimpleFluLocationRule':
        if SimpleFluLocationRule in rule_cls:
            return jsonify({ 'res': False, 'err': 'Rule already in the simulation' })
        sim.add_rule(SimpleFluLocationRule())
        return jsonify({ 'res': True, 'rules': session['sim-flu-ac'].get_state()['rules'] })

    return jsonify({ 'res': False, 'err': 'Rule not supported by this simulation' })

# ----------------------------------------------------------------------------------------------------------------------
def sim_flu_ac_init(session, do_force=False):
    if 'sim-flu-ac' in session and not do_force:
        return

    if 'sim-flu-ac' in session:
        session.pop('sim-flu-ac')

    # site_home = Site('home')
    # school_l  = Site(450149323)  # 88% low income students
    # school_m  = Site(450067740)  #  7% low income students

    session['sim-flu-ac'] = (
        Simulation().
            set().
                pragma_autocompact(True).
                pragma_live_info(True).
                pragma_live_info_ts(False).
                done()
            # add().
                # rule(SimpleFluProgressRule()).
                # rule(SimpleFluLocationRule()).
                # probe(probe_flu_at(school_l, 'low-income')).
                # probe(probe_flu_at(school_m, 'med-income')).
                # done()
    )

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-reset', methods=['GET'])
def sim_flu_ac_reset():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    sim_flu_ac_init(session, True)

    return jsonify({ 'res': True, 'state': session['sim-flu-ac'].get_state() })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-reset-pop', methods=['GET'])
def sim_flu_ac_reset_pop():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    session['sim-flu-ac'].reset_pop()

    return jsonify({ 'res': True, 'state': session['sim-flu-ac'].get_state() })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-reset-probes', methods=['GET'])
def sim_flu_ac_reset_probes():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    session['sim-flu-ac'].reset_probes()

    return jsonify({ 'res': True, 'state': session['sim-flu-ac'].get_state() })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-reset-rules', methods=['GET'])
def sim_flu_ac_reset_rules():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    session['sim-flu-ac'].reset_rules()

    return jsonify({ 'res': True, 'state': session['sim-flu-ac'].get_state() })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-run', methods=['POST'])
def sim_flu_ac_run():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if not 'sim-flu-ac' in session:
        return jsonify({ 'res': False, 'err': 'The simulation has not been initialized' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    if session.get('sim', None) == 'flu-a':
        task_id = session.get('task-id', None)
        task = tasks.get(task_id, None) if task_id else None
        if task and task.state in [states.PENDING, states.RECEIVED, states.STARTED, states.RETRY]:
            return jsonify({ 'res': False, 'err': 'A simulation is in progress' })
        else:
            sim_clear(task_id)

    task = sim_flu_ac_run_bg.apply_async()
    tasks[task.id] = task

    session['sim'] = 'flu-a'
    session['task-id'] = task.id

    return jsonify({ 'res': True }), 202, { 'Location': url_for('task_status', task_id=task.id) }

# ----------------------------------------------------------------------------------------------------------------------
@celery.task(bind=True)
def sim_flu_ac_run_bg(self):
    import time
    n = 10
    for i in range(n):
        time.sleep(0.1)
        self.update_state(state='PROGRESS', meta={ 'i': i, 'n': n, 'p': float(i) / float(n) })

    return { 'i': n, 'n': n, 'p': 1 }

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-status', methods=['GET'])
def sim_flu_ac_status():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None) == 'flu-a':
        return task_status(session.get('task-id', None))

    return jsonify({ 'res': False, 'isRunning': False })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-set-pragma', methods=['POST'])
def sim_flu_ac_set_pragma():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    # if session.get('sim', None):
    #     return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    name  = request.values.get('name',  '', type=str)
    value = sim_get_pragma_request_value(name, request)
    session['sim-flu-ac'].set_pragma(name, value)

    return jsonify({ 'res': True })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-ac-stop', methods=['GET'])
def sim_flu_ac_stop():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None) == 'flu-a':
        sim_clear(session.get('task-id', None))

    return jsonify({ 'res': True })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SIM: TEST ]-----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/sim-test-run', methods=['POST'])
def sim_test_run():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None):
        return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    if session.get('sim', None) == 'test':
        task_id = session.get('task-id', None)
        task = tasks.get(task_id, None) if task_id else None
        if task and task.state in [states.PENDING, states.RECEIVED, states.STARTED, states.RETRY]:
            return jsonify({ 'res': False, 'err': 'A simulation is in progress' })
        else:
            sim_clear(task_id)

    # with app.app_context():
    task = sim_test_run_bg.apply_async()  # queue='transient'
    tasks[task.id] = task

    session['sim'] = 'test'
    session['task-id'] = task.id

    return jsonify({ 'res': True }), 202, { 'Location': url_for('task_status', task_id=task.id) }

# ----------------------------------------------------------------------------------------------------------------------
@celery.task(bind=True)  # name=
def sim_test_run_bg(self):
    import time
    n = 100
    for i in range(n):
        time.sleep(0.1)
        self.update_state(state='PROGRESS', meta={ 'i': i, 'n': n, 'p': float(i) / float(n) })

    # Marking the simulation as done needs to happen in the task_status() function because the current function runs in
    # a separate thread and therefore doesn't have access to the Flask's session object.  I've spent north of 24h on
    # trying to solve this and I'm leaving it here for now.

    return { 'i': n, 'n': n, 'p': 1 }

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-test-status', methods=['GET'])
def sim_test_status():
    # if session.get('sim', None) == 2:
    #     task_id = session.get('task-id', None)
    #     task = tasks.get(task_id, None) if task_id else None
    #     if task and task.state in [states.PENDING, states.RECEIVED, states.STARTED, states.RETRY]:
    #         return jsonify({ 'res': False, 'err': 'A simulation is in progress' })
    #     else:
    #         sim_clear(task_id)
    #         return jsonify({ 'res': True, 'isRunning': False })

    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None) == 'test':
        return task_status(session.get('task-id', None))
    return jsonify({ 'res': False, 'isRunning': False })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-test-stop', methods=['GET'])
def sim_test_stop():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None) == 'test':
        sim_clear(session.get('task-id', None))

    # if session.get('sim', None) == 2:
    #     session.pop('sim', None)
    #     session.pop('task-id', None)
    return jsonify({ 'res': True })
