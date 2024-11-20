from pathlib import Path
import HyperParameters
import Utils as U

#Dataset location information 
PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DATA_FOLDER = (PROJECT_DIR / 'data_clean').resolve()
RAW_DATA_FOLDER = (PROJECT_DIR / 'data_raw').resolve()
TEST_DATA_FOLDER = (PROJECT_DIR / 'data_test').resolve()
MODEL_FOLDER = (PROJECT_DIR / 'Model').resolve()

movie_data = (U.RAW_DATA_FOLDER / 'movies.dat').resolve() 
ratings_data = (U.RAW_DATA_FOLDER / 'ratings.dat').resolve() 
users_data = (U.RAW_DATA_FOLDER / 'users.dat').resolve() 

train_graphs = (U.CLEAN_DATA_FOLDER / 'Processed_Training_Graphs.pt').resolve()
test_graphs = (U.CLEAN_DATA_FOLDER / 'Processed_Testing_Graphs.pt').resolve()
