import importlib
import libsbml
import amici
import os
import sys
sys.path.insert(0, '/data/numerik/people/abankowski/neuro_paramestimation/src')

# name of SBML file
sbml_file = "recovery_model_60_peaks_qssa_rates.xml"
# name of the model that will also be the name of the python module
model_name = "recovery_model_60_peaks_qssa_rates" 
#directory to which the generated model code is written
model_output_dir = "recovery_model_60_peaks_qssa_rates"#model_name


"""
If this is the first time running the code and there is no recovery_model_60_peaks_qssa_rates python module, run this to import it with amici 
"""
if __name__ == "__main__":
    sbml_reader = libsbml.SBMLReader() #blub
    sbml_doc = sbml_reader.readSBML(sbml_file)
    sbml_model = sbml_doc.getModel()
    #import sbml model, compile and generate amici module
    sbml_importer = amici.SbmlImporter(sbml_file)
    sbml_importer.sbml2amici(model_name, model_output_dir, verbose=True,generate_sensitivity_code = False)