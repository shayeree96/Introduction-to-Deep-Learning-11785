import numpy as np

import json
import traceback

def assertions_all(user_vals, expected_vals, test_name, rtol=1e-5, atol=1e-8):
    if not assertions(user_vals, expected_vals, 'type', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'shape', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'closeness', test_name, rtol=rtol, atol=atol):
        return False
    return True

def assertions(user_vals, expected_vals, test_type, test_name, rtol=1e-5, atol=1e-8):
    if test_type == 'type':
        try:
            assert type(user_vals) == type(expected_vals)
        except Exception as e:
            print('Type error, your type doesnt match the expected type.')
            print('Wrong type for %s' % test_name)
            print('Your type:   ', type(user_vals))
            print('Expected type:', type(expected_vals))
            return False
    elif test_type == 'shape':
        try:
            assert user_vals.shape == expected_vals.shape
        except Exception as e:
            print('Shape error, your shapes doesnt match the expected shape.')
            print('Wrong shape for %s' % test_name)
            print('Your shape:    ', user_vals.shape)
            print('Expected shape:', expected_vals.shape)
            return False
    elif test_type == 'closeness':
        try:
            assert np.allclose(user_vals, expected_vals, rtol=rtol, atol=atol)
        except Exception as e:
            print('Closeness error, your values dont match the expected values.')
            print('Wrong values for %s' % test_name)
            print('Your values:    ', user_vals)
            print('Expected values:', expected_vals)
            return False
    return True

def print_failure(cur_test, num_dashes=51):
    print('*' * num_dashes)
    print('The local autograder will not work if you do not pass %s.' % cur_test)
    print('*' * num_dashes)
    print(' ')

def print_name(cur_question):
    print(cur_question)

def print_outcome(short, outcome, point_value, num_dashes=51):
    score = point_value if outcome else 0
    if score != point_value:
        print("{}: {}/{}".format(short, score, point_value))
        print('-' * num_dashes)

def run_tests(tests, summarize=False):
    # calculate number of dashes to print based on max line length
    title = "AUTOGRADER SCORES"
    num_dashes = calculate_num_dashes(tests, title)

    # print title of printout
    print(generate_centered_title(title, num_dashes))
    
    # Print each test
    scores = {}
    for t in tests:
        if not summarize:
            print_name(t['name'])
        try:
            res = t['handler']()
        except Exception:
            res = False
            traceback.print_exc()
        if not summarize:
            print_outcome(t['autolab'], res, t['value'], num_dashes)
        scores[t['autolab']] = t['value'] if res else 0

    points_available = sum(t['value'] for t in tests)
    points_gotten = sum(scores.values())
    print("Total score: {}/{}\n".format(points_gotten, points_available))

    print("Summary:")
    print(json.dumps({'scores': scores}))

def calculate_num_dashes(tests, title):
    """Determines how many dashes to print between sections (to be ~pretty~)"""
    # Init based on str lengths in printout
    str_lens = [len(t['name']) for t in tests] + [len(t['autolab']) + 4 for t in tests]
    num_dashes = max(str_lens) + 1

    # Guarantee minimum 5 dashes around title
    if num_dashes < len(title) - 4:
        return len(title) + 10

    # Guarantee even # dashes around title
    if (num_dashes - len(title)) % 2 != 0:
        return num_dashes + 1

    return num_dashes

def generate_centered_title(title, num_dashes):
    """Generates title string, with equal # dashes on both sides"""
    dashes_on_side = int((num_dashes - len(title)) / 2) * "-"
    return dashes_on_side + title + dashes_on_side

def save_numpy_array(np_array, file_name):
    with open(file_name, 'wb') as f:
        np.save(f, np_array)

def load_numpy_array(file_path):
    with open(file_path, 'rb') as f:
        output = np.load(f, allow_pickle=True)
    return output