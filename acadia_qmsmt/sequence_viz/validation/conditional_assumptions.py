"""Unresolved scalar comparisons must describe one consistent possible value.

The compiler's ==/!= and </>= tests are mutually exclusive for an unchanged
register. The viewer may label a placeholder as assumed, but cannot run both
bodies by assuming each comparison true independently. No hardware required.
"""
from acadia_qmsmt.sequence_viz.tracing import Block, SequenceTrace

pairs = [('==', '!='), ('<', '>='), ('>', '<=')]
checked = 0
for first, second in pairs:
    for literal in (-1, 0, 1):
        conditions = [f'REG0 {operator} {literal}' for operator in (first, second)]
        trace = SequenceTrace(blocks=[
            Block(index=j, start=0, length=1, trigger=False, blocking=False,
                  conditional=({'kind':'test', 'condition':condition, 'id':j},))
            for j, condition in enumerate(conditions)])
        assert all(trace.evaluate_condition(condition) is None for condition in conditions)
        assumed = [trace.evaluate_condition(condition, assume_unknown=True) for condition in conditions]
        assert sum(assumed) == 1, (conditions, assumed)
        trace.assumed_constructs = set()  # normally initialized by relayout
        plan = trace.execution_plan()
        assert len(plan) == 1 and plan[0][0] == assumed.index(True), (conditions, plan)
        assert len(trace.assumed_constructs) == 2  # both comparisons remain labelled assumed
        for known in (-2, 0, 2):
            trace.register_cycles['REG0'] = known
            expected = [eval(condition.replace('REG0', str(known))) for condition in conditions]
            assert [trace.evaluate_condition(condition, assume_unknown=True) for condition in conditions] == expected
        checked += 1
assert checked == 9
print('COMPARED 9 complementary condition pairs: one default body, both remain assumed, 27 known-value pairs preserved')

trace = SequenceTrace(blocks=[
    Block(index=0,start=0,length=1,trigger=False,blocking=False,
          conditional=({'kind':'test','condition':'REG0 != 0','id':0},)),
    Block(index=1,start=0,length=1,trigger=False,blocking=False,
          conditional=({'kind':'test','condition':'REG0 == 0','id':1},))])
trace.assumed_constructs = set()
for first in (False, True):
    trace.path_choices = {trace.construct_key(0,1):first,trace.construct_key(1,1):not first}
    assert [entry[0] for entry in trace.execution_plan()] == ([0] if first else [1])
print('COMPARED 2 explicit alternative paths: pins retain priority')
