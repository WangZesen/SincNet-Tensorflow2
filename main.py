import os
import logging
import argparse
import importlib
import subprocess

def get_recipes():
    recipes = os.listdir(os.path.join(os.path.dirname(__file__), 'source'))
    recipes = filter(lambda x: x[0].isalpha() and x.endswith('.py'), recipes)
    recipes = map(lambda x: x[:-3], recipes)
    return list(recipes)

logger = logging.getLogger()

if __name__ == '__main__':
    recipes = get_recipes()

    parser = argparse.ArgumentParser(
        description='SincNet Keyword Spotting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-r', '--recipe', type=str, help=f'recipe name, {recipes}')
    parser.add_argument('-c', '--config', type=str, help='configure file')
    parser.add_argument('-t', '--template', type=str, help='print template file')

    args = parser.parse_args()
    
    if not (args.recipe in recipes):
        logger.error(f'invalid recipe name: {args.recipe}. available: {recipes}')
        exit(0)
    
    module = importlib.import_module('source.' + args.recipe)

    if args.template:
        module.get_template(args.template)
    
    if args.config:
        module.process(args.config)
    
    