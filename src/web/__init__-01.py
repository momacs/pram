from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('base.html')

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return jsonify('pong')

# # ----------------------------------------------------------------------------------------------------------------------
# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     a = request.values.get('a', 0, type=float)
#     b = request.values.get('b', 0, type=float)
#     return jsonify(res = a + b)

# ----------------------------------------------------------------------------------------------------------------------
@app.route('/sim01', methods=['GET', 'POST'])
def sim01():
    from pram.data   import GroupSizeProbe, ProbeMsgMode
    from pram.entity import Site
    from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, TimePoint
    from pram.sim    import Simulation

    sites = { s:Site(s) for s in ['home', 'school-a', 'school-b']}

    probe_grp_size_site = GroupSizeProbe.by_rel('site', Site.AT, sites.values(), msg_mode=ProbeMsgMode.CUMUL)

    (Simulation().
        add().
            rule(ResetSchoolDayRule(TimePoint(7))).
            rule(GoToAndBackTimeAtRule(t_at_attr='t@school')).
            probe(probe_grp_size_site).
            commit().
        new_group(500).
            set_rel(Site.AT,  sites['home']).
            set_rel('home',   sites['home']).
            set_rel('school', sites['school-a']).
            commit().
        new_group(500).
            set_rel(Site.AT,  sites['home']).
            set_rel('home',   sites['home']).
            set_rel('school', sites['school-b']).
            commit().
        run(18)
    )

    return probe_grp_size_site.get_msg()
