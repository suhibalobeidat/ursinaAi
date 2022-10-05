import argparse
from teachDRL.teachers.teacher_controller import TeacherController
from collections import OrderedDict
import numpy as np

def get_teacher():

    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher', type=str, default="ALP-GMM")  # ALP-GMM, Covar-GMM, RIAC, Oracle, Random
    parser.add_argument('--nb_test_episodes', type=int, default=50)
    parser.add_argument('--seed', '-s', type=int, default=0)

    parser.add_argument('--gmm_fitness_fun', '-fit', type=str, default=None)
    parser.add_argument('--nb_em_init', type=int, default=None)
    parser.add_argument('--min_k', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=None)
    parser.add_argument('--fit_rate', type=int, default=None)
    parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
    parser.add_argument('--alp_max_size', type=int, default=None)

    args = parser.parse_args()

    params = {}
    if args.teacher == 'ALP-GMM':
        if args.gmm_fitness_fun is not None:
            params['gmm_fitness_fun'] = args.gmm_fitness_fun
        if args.min_k is not None and args.max_k is not None:
            params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
        if args.weighted_gmm is True:
            params['weighted_gmm'] = args.weighted_gmm
        if args.nb_em_init is not None:
            params['nb_em_init'] = args.nb_em_init
        if args.fit_rate is not None:
            params['fit_rate'] = args.fit_rate
        if args.alp_max_size is not None:
            params['alp_max_size'] = args.alp_max_size
    elif args.teacher == 'Covar-GMM':
        if args.absolute_lp is True:
            params['absolute_lp'] = args.absolute_lp
    elif args.teacher == "RIAC":
        if args.max_region_size is not None:
            params['max_region_size'] = args.max_region_size
        if args.alp_window_size is not None:
            params['alp_window_size'] = args.alp_window_size


    param_env_bounds = OrderedDict()

    #param_env_bounds['total_distance'] = [10, 25]
    param_env_bounds['total_distance'] = [0, 1,2]

    #param_env_bounds['number of spaces away from start point'] = [0,5]





    Teacher = TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds,
                                seed=args.seed, teacher_params=params)
    return Teacher