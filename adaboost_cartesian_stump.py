from math import exp, log
from typing import Iterable, List, Literal, Tuple


class Point:
    def __init__(self, val: Literal[1, -1], wt: float, label: str, *coordinates: float) -> None:
        self.coordinates = coordinates
        self.val = val
        self.wt = wt
        self.label = label

    def __repr__(self) -> str:
        return f'{self.label}: {self.wt}'


class State:
    def __init__(self) -> None:
        self.min_dir: int = None
        '''Index of the coordinate axis for which error is minimum'''

        self.min_idx: int = None
        '''Index of the point on `min_dir` at which split occurs'''

        self.min_val = float('inf')
        '''ε'''

        self.plus_ax: Literal[1, -1] = None
        '''1: aligns with axis direction, plus is towards +ve infinity
        -1: opposite axis direction, plus is towards -ve infinity'''

        self.misclassified: List[Point] = []
        '''List of misclassified points after split'''

    def __repr__(self) -> str:
        inf_dir = 'positive' if self.plus_ax == 1 else 'negative'
        return f'Current best split: Axis index {self.min_dir} at {self.min_idx}; positive class towards {inf_dir} infinity with ε = {self.min_val}; Misclassified: {self.misclassified}'


def split_best_for_axis(points: List[Point], dim_idx: int, state: State):
    ascending = sorted(points, key=lambda point: point.coordinates[dim_idx])
    prev = None

    # calculate at lower edge
    plus_to_axis_plus = sum(map(lambda point: point.wt, [
        point for point in ascending if point.val == -1]))
    plus_to_axis_minus = sum(map(lambda point: point.wt, [
        point for point in ascending if point.val == 1]))
    if plus_to_axis_minus < plus_to_axis_plus and state.min_val > plus_to_axis_minus:
        state.plus_ax = -1
        state.min_val = plus_to_axis_minus
        state.min_dir = dim_idx
    elif state.min_val > plus_to_axis_plus:
        state.plus_ax = 1
        state.min_val = plus_to_axis_plus
        state.min_dir = dim_idx

    for point_idx in range(len(ascending)):
        point = ascending[point_idx]
        if point.coordinates[dim_idx] == prev:
            continue
        prev = point.coordinates[dim_idx]
        plus_to_axis_plus = sum(map(lambda pt: pt.wt, [
            pt for pt in ascending if pt.val == -1 and pt.coordinates[dim_idx] > point.coordinates[dim_idx]])) + sum(map(lambda pt: pt.wt, [
                pt for pt in ascending if pt.val == 1 and pt.coordinates[dim_idx] <= point.coordinates[dim_idx]]))
        plus_to_axis_minus = sum(map(lambda pt: pt.wt, [
            pt for pt in ascending if pt.val == 1 and pt.coordinates[dim_idx] > point.coordinates[dim_idx]])) + sum(map(lambda pt: pt.wt, [
                pt for pt in ascending if pt.val == -1 and pt.coordinates[dim_idx] <= point.coordinates[dim_idx]]))
        if plus_to_axis_minus < plus_to_axis_plus and state.min_val > plus_to_axis_minus:
            state.plus_ax = -1
            state.min_val = plus_to_axis_minus
            state.min_idx = point.coordinates[dim_idx] + 0.5
            state.min_dir = dim_idx
            state.misclassified = [
                pt for pt in ascending if pt.val == 1 and pt.coordinates[dim_idx] > point.coordinates[dim_idx]] + [
                pt for pt in ascending if pt.val == -1 and pt.coordinates[dim_idx] <= point.coordinates[dim_idx]]
        elif state.min_val > plus_to_axis_plus:
            state.plus_ax = 1
            state.min_val = plus_to_axis_plus
            state.min_idx = point.coordinates[dim_idx] + 0.5
            state.min_dir = dim_idx
            state.misclassified = [
                pt for pt in ascending if pt.val == -1 and pt.coordinates[dim_idx] > point.coordinates[dim_idx]] + [
                pt for pt in ascending if pt.val == 1 and pt.coordinates[dim_idx] <= point.coordinates[dim_idx]]


def get_alpha(misclassified: List[Point]):
    tot_wt = sum(map(lambda x: x.wt, misclassified))
    alpha = 0.5 * log((1 - tot_wt) / tot_wt)
    return alpha


def adjust_wt_correct(points: List[Point], alpha: float):
    for point in points:
        point.wt *= exp(-alpha)


def adjust_wt_incorrect(points: List[Point], alpha: float):
    for point in points:
        point.wt *= exp(alpha)


def normalize(points: List[Point]):
    tot_wt = sum(map(lambda x: x.wt, points))
    print(f'z = {tot_wt}')
    for point in points:
        point.wt /= tot_wt


def predict(model: Iterable[Tuple[float, State]], to_classify: Tuple[float, ...]):
    result = 0
    for eq in model:
        alpha, state = eq
        relevant_measure = to_classify[state.min_dir]
        result += (alpha * state.plus_ax *
                   (1 if relevant_measure > state.min_idx else -1))
    return result
