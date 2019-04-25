#
# Next
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
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Dev
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
#
# Prod (FreeBSD 12R)
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

import os
import sys
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # pram pkg path


import os
import psutil
import shutil

import config

from celery import Celery, states
from celery.task.control import revoke
from flask import Flask, current_app, jsonify, request, render_template, Response, session, url_for
from flask_session import Session
# from flask.ext.session import Session

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
from pram.sim    import Simulation


SUDO_CODE = 'catch22'  # TODO: Use env variable
LOAD_CPU_INT = 1  # CPU load sample interval [s]

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

def bytes2human(b):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}

    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10

    for s in reversed(symbols):
        if b >= prefix[s]:
            value = float(b) / prefix[s]
            return '{:.1f}{}'.format(value, s)

    return '{}B'.format(b)


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SYS ]-----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
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
            'ram': f'{bytes2human(psutil.virtual_memory().total)}',
            'hdd': f'{bytes2human(hdd_size)}'
        },
        'used': {
            'cpu': f'{cpu_used}%',
            'ram': f'{bytes2human(psutil.virtual_memory().used)}',
            'hdd': f'{bytes2human(hdd_used)} ({round(float(hdd_used) / float(hdd_size) * 100, 0)}%)'
        },
        'free': {
            'cpu': f'{100 - cpu_used}%',
            'ram': f'{bytes2human(psutil.virtual_memory().available)}',
            'hdd': f'{bytes2human(hdd_free)} ({round(float(hdd_free) / float(hdd_size) * 100, 0)}%)'
        }
    })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sys-ping', methods=['HEAD'])
def sys_ping():
    res = Response()
    res.headers.add('content-length', 1)
    return res

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-clr-sess', methods=['GET'])
def usr_clr_sess():
    session.clear()
    return jsonify({ 'res': True})

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-get-sess', methods=['GET'])
def usr_get_sess():
    return jsonify({ 'sim': session.get('sim', None) })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-is-root', methods=['GET'])
def usr_is_root():
    ''' Check if the user has elevated access right. '''

    return jsonify(session.get('is-root', False))

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/usr-toggle', methods=['POST'])
def usr_toggle():
    ''' Grant the user elevated access rights if the code submitted is correct. '''

    code = request.values.get('code', '', type=str)
    session['is-root'] = (code == SUDO_CODE)
    return jsonify(session['is-root'])


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
# ----[ SIM: FLU ]------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/sim-flu-reset', methods=['GET'])
def sim_flu_reset():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None):
        return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    if 'sim-flu' in session:
        del session['sim-flu']
    return jsonify({ 'res': True })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-run', methods=['POST'])
def sim_flu_run():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None):
        return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    iter_n  = request.values.get('iter-n',  '', type=int)
    grp_a_n = request.values.get('grp-a-n', '', type=int)
    grp_b_n = request.values.get('grp-b-n', '', type=int)

    if not session.get('sim-flu', None):
        sites = { s:Site(s) for s in ['home', 'school-a', 'school-b']}
        probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.CUMUL)

        session['sim-flu'] = (
            Simulation().
                add().
                    rule(ResetSchoolDayRule(TimePoint(7))).
                    rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
                    probe(probe_grp_size_site).
                    commit().
                new_group(grp_a_n).
                    set_rel(Site.AT,  sites['home']).
                    set_rel('home',   sites['home']).
                    set_rel('school', sites['school-a']).
                    commit().
                new_group(grp_a_n).
                    set_rel(Site.AT,  sites['home']).
                    set_rel('home',   sites['home']).
                    set_rel('school', sites['school-b']).
                    commit()
        )

    sim = session.get('sim-flu', None)
    probe = sim.probes[0]
    probe.clear()
    sim.run(iter_n)

    return jsonify({ 'res': True, 'out': probe.get_msg() })


# ----------------------------------------------------------------------------------------------------------------------
# ----[ SIM: FLU ALLEGHENY ]--------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

@app.route('/sim-flu-a-run', methods=['POST'])
def sim_flu_a_run():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None):
        return jsonify({ 'res': False, 'err': 'A simulation is in progress' })

    if session.get('sim', None) == 'flu-a':
        task_id = session.get('task-id', None)
        task = tasks.get(task_id, None) if task_id else None
        if task and task.state in [states.PENDING, states.RECEIVED, states.STARTED, states.RETRY]:
            return jsonify({ 'res': False, 'err': 'A simulation is in progress' })
        else:
            sim_clear(task_id)

    task = sim_flu_a_run_bg.apply_async()
    tasks[task.id] = task

    session['sim'] = 'flu-a'
    session['task-id'] = task.id

    return jsonify({ 'res': True }), 202, { 'Location': url_for('task_status', task_id=task.id) }

# ----------------------------------------------------------------------------------------------------------------------
@celery.task(bind=True)
def sim_flu_a_run_bg(self):
    import time
    n = 10
    for i in range(n):
        time.sleep(0.1)
        self.update_state(state='PROGRESS', meta={ 'i': i, 'n': n, 'p': float(i) / float(n) })

    return { 'i': n, 'n': n, 'p': 1 }

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-a-status', methods=['GET'])
def sim_flu_a_status():
    if not session.get('is-root', False):
        return jsonify({ 'res': False, 'err': 'Insufficient rights' })

    if session.get('sim', None) == 'flu-a':
        return task_status(session.get('task-id', None))
    return jsonify({ 'res': False, 'isRunning': False })

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim-flu-a-stop', methods=['GET'])
def sim_flu_a_stop():
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
