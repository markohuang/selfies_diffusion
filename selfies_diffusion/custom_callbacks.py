import tempfile
from openbabel import pybel
from PIL import Image
from .selfies_grammar import grammar
from pytorch_lightning.callbacks import Callback
import rdkit.Chem as Chem
pybel.ob.obErrorLog.StopLogging()

class GenerateOnValidationCallback(Callback):
   def __init__(self, generated_fname) -> None:
       super().__init__()
       self.generated_fname = generated_fname

   def on_validation_end(self, trainer, pl_module):       
        wandb_logger = trainer.logger
        selfies_list = []
        for _ in range(9):
            selfies_list.append(pl_module.denoise_sample(1, 100, skip_special_tokens=True)[0])
        wimg_list = []
        for _, selfies in enumerate(selfies_list):
            try:
                smiles = Chem.MolToSmiles(grammar.decoder(selfies.replace(' ', '')))
                mol = pybel.readstring('smi', smiles)
                with tempfile.NamedTemporaryFile(suffix='png', delete=True) as tmp_file:
                    mol.draw(show=False, filename=tmp_file.name)
                    pil_img = Image.open(tmp_file.name)
                    wimg_list.append(pil_img)
            except:
                continue
        if wimg_list:
            wandb_logger.log_image(key='diffused_selfies',images=wimg_list)
