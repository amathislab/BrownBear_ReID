import wandb
import os

class WandbLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        output_dir = self.cfg.OUTPUT_DIR
        dataset_name = output_dir.split('/')[-1]  ### test_on_20xx
        self.project_name = output_dir.split('/')[-3] + '_' + dataset_name

    def init_wandb(self, project_name, model_name, config, notes=None):
        self.run = wandb.init(project=project_name, name=model_name, config=config, notes=notes)
        self.run_id = wandb.run.id
        return self.run

    def resume(self, run_id, project_name=None, resume='allow'):
        if project_name is None:
            project_name = self.project_name
        self.run = wandb.init(project=project_name, resume=resume, id=run_id)
        return self.run


    def build_config(self, cfg, config_file, exp_name, project_name, output_dir):
        model_weights_path = cfg.TEST.WEIGHT
        if model_weights_path is None or model_weights_path == '':
            model_weights_path = os.path.join(cfg.OUTPUT_DIR, 'net_last.pth')   

        config={
            "exp_name": exp_name,
            "project_name": project_name,
            "output_dir": output_dir,
            "config_file": config_file,
            "with_pose": cfg.MODEL.AGG_POSE_FEATURE,
            "learning_rate": cfg.SOLVER.BASE_LR,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "checkpoint": model_weights_path,
            "num_gpus": cfg.MODEL.DEVICE_ID,
            "optimizer": cfg.SOLVER.OPTIMIZER_NAME,
            "epochs": cfg.SOLVER.MAX_EPOCHS,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "momentum": cfg.SOLVER.MOMENTUM,
            "gamma": cfg.SOLVER.GAMMA,
            "output_dir": cfg.OUTPUT_DIR,
        }

        return config

    def update_config(self, config):
        self.run.config.update(config)

    def log_csv_table(self, df, csv_file_path, log_name):
        table = wandb.Table(dataframe=df)
        
        table_artifact = wandb.Artifact(log_name, type="dataset")
        table_artifact.add(table, log_name)

        # Log the raw csv file within an artifact to preserve our data
        table_artifact.add_file(csv_file_path)
        self.run.log({log_name: table})
        # # and Log as an Artifact to increase the available row limit!
        self.run.log_artifact(table_artifact)
    
    def log_image(self, img, log_name):
        self.run.log({log_name: img})
    

    def save_images(self, img_list, log_name):
        self.run.log({log_name: [wandb.Image(img) for img in img_list]})