import pyAgrum as gum
import pyAgrum.lib.image as gumimage
from IPython.display import Image

def save_bdn_diagram(bdn_model, filename="bdn_diagram", file_format="png"):
    """
    Display and save a Bayesian Decision Network (BDN) influence diagram.

    Parameters
    ----------
    bdn_model : gum.InfluenceDiagram
        The BDN model to be visualized and saved.
    filename : str, optional
        The name of the file to save the diagram to. Default is "bdn_diagram.png".
    file_format : str, optional
        The format to save the file in (e.g., "png", "pdf", "svg"). Default is "png".
    """
    gumimage.export(bdn_model,f'{filename}.{file_format}')
    Image(filename=f'{filename}.{file_format}')