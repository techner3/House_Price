import os 
from pathlib import Path

project_name = "house_price"

list_of_files=[f"{project_name}/__init__.py",
               f"{project_name}/components/__init__.py",
               f"{project_name}/components/data_ingestion.py",
               f"{project_name}/components/data_validation.py",
               f"{project_name}/components/data_transformation.py",
               f"{project_name}/components/model_trainer.py",
               f"{project_name}/components/model_pusher.py",
               f"{project_name}/constants/__init__.py",
               f"{project_name}/exception/__init__.py",
               f"{project_name}/logger/__init__.py"
               f"{project_name}/utils/__init__.py",
               f"{project_name}/entity/artifact_entity.py",
               f"{project_name}/entity/config_entity.py",
               f"requirements.py",
               f"setup.py",
               f"app.py"]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir!="":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
    else:
        print(f"{filepath} already exists !")