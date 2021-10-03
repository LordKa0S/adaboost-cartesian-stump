from adaboost_cartesian_stump import Point, State, adjust_wt_correct, adjust_wt_incorrect, get_alpha, normalize, predict, split_best_for_axis

if __name__ == "__main__":
    # execute only if run as a script

    max_iter = 4
    '''Number of boosting iterations'''

    init_wt = 1/10
    '''Initial weight for each sample `Point`'''

    A1 = Point(1, init_wt, 'a1', 2, 3)
    A2 = Point(1, init_wt, 'a2', 2, 2)
    A3 = Point(1, init_wt, 'a3', 4, 6)
    A4 = Point(1, init_wt, 'a4', 5, 7)
    A5 = Point(1, init_wt, 'a5', 6, 5)
    B1 = Point(-1, init_wt, 'b1', 4, 3)
    B2 = Point(-1, init_wt, 'b2', 4, 1)
    B3 = Point(-1, init_wt, 'b3', 5, 3)
    B4 = Point(-1, init_wt, 'b4', 8, 6)
    B5 = Point(-1, init_wt, 'b5', 8, 2)

    points = [A1, A2, A3, A4, A5, B1, B2, B3, B4, B5]
    '''List of labelled sample `Point`s'''

    model = []
    for boost_iter in range(max_iter):
        print(f'Iteration: {boost_iter + 1}')
        print(f'Initial: {points}')
        state = State()
        for dim_idx in range(len(points[0].coordinates)):
            split_best_for_axis(points, dim_idx, state)
        print(state)
        alpha = get_alpha(state.misclassified)
        print(f'α = {alpha}')
        model.append((alpha, state))
        adjust_wt_incorrect(state.misclassified, alpha)
        adjust_wt_correct(
            [point for point in points if point not in state.misclassified], alpha)
        print(f'Pre-normalize: {points}')
        normalize(points)
        print(f'Final: {points}')

    to_classify = (7, 2)
    '''Sample to be labelled'''

    print(predict(model, to_classify))
