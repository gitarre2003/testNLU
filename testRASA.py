# NLU Model Training
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter

# train the model
# data: path to the training data
# config: path to the pipeline (https://rasa.com/docs/nlu/choosing_pipeline/)
# model_dir: path to the models. By default, a path "<model_dir>/default/<fixed_model_name>" is created.
#            <fixed_model_name> is also the project name
def train_nlu(data, configs, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = 'current')

# use the model
def run_nlu():
    interpreter = Interpreter.load('./models/default/current/')
    print(interpreter.parse("Can I book an appointment please?"))

# run the program
if __name__ == '__main__':
    train_nlu('data/demo-rasa.json', "nlu_config.yml", 'models/')
    run_nlu()
    
# TO RUN as a HTTP Server:
# 1. cd to "./" folder
# 2. run "python -m rasa_nlu.server --path <model_dir>/default"
# 3. In browser parse: http://localhost:5000/parse?q=<dialog>&project=<fixed_model_name>

