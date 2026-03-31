# Electrode Mechanics (Sn Anode Mechanics for Li-ion Batteries)
# Informatics of Electrode Mechanics 

# Named Entity Recognition for Linking Text (Entity) to Knowledge Base of Li-ion Battery with Sn Anode (Supervised NLP)

Basic Model (deselection of irrelevant phase parameters, suitable for smaller text database)
[![meaningtowords](https://img.shields.io/badge/supervisedNER-streamlit-red)](https://supervised-nlp-electrodephases.streamlit.app/ )

Advanced Model (deselection of irrelevant phase parameters, suitable for smaller text database)
[![meaningtowords](https://img.shields.io/badge/supervisedAdvancedNERsmall-streamlit-red)](https://advancednlpelectrodephases.streamlit.app/ )


Advanced Model (selection of relevant phase parameters, suitable for larger text database)
[![meaningtowords](https://img.shields.io/badge/supervisedAdvancedNERlarge-streamlit-red)](https://electrodephasemechanics-advancednlp.streamlit.app/ )

Advanced Model (selection of relevant phase parameters, suitable for larger text database. Quantitative representation of Bibliometric Network)
[![meaningtowords](https://img.shields.io/badge/supervisedQNetNERlarge-streamlit-red)](https://electrodemechanics-nlpquantitativenetwork.streamlit.app/ )

Illustration of Hybrid NER (without the Pie-Chart) 
[![meaningtowords](https://img.shields.io/badge/hybridNER-streamlit-red)](https://hybrid-ner-in-anodemechanics.streamlit.app/)

Illustration of Hybrid NER (with the Pie-Chart) 
[![meaningtowords](https://img.shields.io/badge/hybridNERwPieChart-streamlit-red)](https://hybrid-ner-anode-mechanics.streamlit.app/)

# Intelligent knowledge extraction for understanding volume expansion in lithiated battery electrode (Attention Mechanism, NLP)

NER analysis for terms such as : Battery (specific capacity), Volume expansion, Mechanical Stress and Strain. 

A. Intelligent app to study the text of Lithium ion battery mechanics and store them in database format: lithiation_knowledge.db (concise information) and knowledge_universe.db (full knowledge). Web app (to be made available soon...)

Relevance score and Attention Mechanism:
[![meaningtowords](https://img.shields.io/badge/relScore-streamlit-red)](https://relevance-score-elastomechanics.streamlit.app/)

More interactive relevance score (Heatmap)
[![meaningtowords](https://img.shields.io/badge/interactiverelScore-streamlit-red)](https://interactive-relevance-score-elastomechanics.streamlit.app/)

B. Learner tool that reads the .db files through the following stepwise processes

(i) inspect the knowledge databases and provide the key summary

(ii) common term analysis in diverse literature 

(iii) NER analysis for the most significant words/phrases summarized from (i)-(ii)

Lithium ion Battery Mechanics Learner app (Basic model): 
[![meaningtowords](https://img.shields.io/badge/attnMechLearner-streamlit-red)](https://lithiumionbatterymechanics-learner.streamlit.app/)

If the basic model fails to run (due to memory issues) then the following visalization tools can be utilized to see only the results:
Basic Visualization
[![meaningtowords](https://img.shields.io/badge/attnMechVisual-streamlit-red)](https://nlpsimulationenergymechanics-datavisualization.streamlit.app/)

Enhanced Visualization 
[![meaningtowords](https://img.shields.io/badge/attnMechEnhancedVisual-streamlit-red)](https://energymehanics-informatics-enhanceddatavisualization.streamlit.app/)

# Informatics based quantification of interface energy (J/m2) and diffuse interface width (nm) for phase field model

[![meaningtowords](https://img.shields.io/badge/nerInterface-streamlit-red)](https://interface-attributes-ner.streamlit.app/)


# Phase Field Model

Interpolation Function

Arctan function for h : 
[![meaningtowords](https://img.shields.io/badge/arctanfunction-streamlit-red)](https://arctaninterpolationfunction.streamlit.app/)

Sigmoidal : 
[![meaningtowords](https://img.shields.io/badge/sigmoidal-streamlit-red)](https://sigmoidalinterpolationfunction.streamlit.app/)

Smooth Function and Sigmoidal: 
[![meaningtowords](https://img.shields.io/badge/sigmalog-streamlit-red)](https://sigmoidallogarithmic-interpolationfunction.streamlit.app/)

Hyperbolic Tangent:
[![meaningtowords](https://img.shields.io/badge/tanh-streamlit-red)](https://hyperbolictangent-interpolationfunction.streamlit.app/)

Multiple Functions Model for Interpolating Material Properties of BCT Sn and LI2SN5 Phases
[![meaningtowords](https://img.shields.io/badge/multiplefunc-streamlit-red)](https://multimodelinterpolationfunction.streamlit.app/)


Eigen Strain 

Eigenstrain Calculation from the Molar Volume of Phases:
Two Models
[![meaningtowords](https://img.shields.io/badge/eigenstrain-streamlit-red)](https://eigenstraincalculator.streamlit.app/)

Three Models
[![meaningtowords](https://img.shields.io/badge/phaseeigenstrain-streamlit-red)](https://compute-eigenstrain.streamlit.app/)


# DFT Computation
 The cloud version only run demo calculations, the dft computations has to be performed with the code in local computers

Structural Optimization

[![atomisticmechanics](https://img.shields.io/badge/vcrelaxlisnqe-streamlit-red)](https://structural-optimization-lithium-tin.streamlit.app/)  (This app is dependent on Quantum Espresso, and runs only in local computer)

[![atomisticmechanics](https://img.shields.io/badge/vcrelaxlisngpaw-streamlit-red)](https://structural-optimization-lithium-tin2.streamlit.app/) (runs in cloud environment, ASE uses GPAW calculator, BFGS optimizes only atomic positions inside a fixed cell )

[![atomisticmechanics](https://img.shields.io/badge/vcrelaxlisngpaw2-streamlit-red)](https://structural-optimization-lithium-tin3.streamlit.app/) (runs in cloud environment, ASE uses GPAW calculator, UnitCellFilter allows the unit cell vectors to change during optimization, data storage in sqlite db )

Volume Expansion During Lithiation (vc-relaxation method for speed, and EOS mapping method for accuracy)


[![atomisticmechanics](https://img.shields.io/badge/deltavolumebasic-streamlit-red)](https://volume-expansion-lithium-tin.streamlit.app/) (vc-relaxation method, runs in cloud environment, ASE uses GPAW calculator, uses BGFS for optimization in the fixed cell)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer1-streamlit-red)](https://volume-expansion-lithium-tin2.streamlit.app/) (vc-relaxation method, runs in cloud environment, ASE uses GPAW calculator, uses LBGFS for speedy optimization in the fixed cell)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer2-streamlit-red)](https://volume-expansion-lithium-tin3.streamlit.app/)  (athermal EOS mapping method, only Energy minimization)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer4-streamlit-red)](https://volume-expansion-lithium-tin4.streamlit.app/) (vc-relaxation method, runs in cloud environment, ASE uses GPAW calculator, uses LBGFS and modernized ExpCellFilter for speedy optimization )


[![atomisticmechanics](https://img.shields.io/badge/deltavolumer5-streamlit-red)](https://volume-expansion-athermal-eosmapping5.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer6-streamlit-red)](https://volume-expansion-athermal-eosmapping6.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer7-streamlit-red)](https://volume-expansion-athermal-eosmapping7.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer8-streamlit-red)](https://volume-expansion-athermal-eosmapping8.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, fast)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer9-streamlit-red)](https://volume-expansion-athermal-eosmapping9.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures,DEMO mode, fast and solutions available)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer10-streamlit-red)](https://volume-expansion-athermal-eosmapping10.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, fast and robust, phase 1 successful, others still in testing phase )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer11-streamlit-red)](https://volume-expansion-athermal-eosmapping11.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer12-streamlit-red)](https://volume-expansion-athermal-eosmapping12.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer13-streamlit-red)](https://volume-expansion-athermal-eosmapping13.streamlit.app/ )  (athermal EOS mapping method,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer14-streamlit-red)](https://volume-expansion-athermal-eosmapping14.streamlit.app/ )  (athermal EOS mapping method,  E-V mapping works,   Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer15-streamlit-red)](https://volume-expansion-athermal-eosmapping15.streamlit.app/ )  (athermal EOS mapping method, E-V mapping works,    Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, phase 1 computation works )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer16-streamlit-red)](https://volume-expansion-athermal-eosmapping16.streamlit.app/ )  (athermal EOS mapping method, E-V mapping works and correct for both beta Sn and Li2Sn5 ,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases except phase 1 robust )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer17-streamlit-red)](https://volume-expansion-athermal-eosmapping17.streamlit.app/ )  (athermal EOS mapping method, E-V mapping works and is incorrect for beta Sn while it is correct for Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization with incorrect output  )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer18-streamlit-red)](https://volume-expansion-athermal-eosmapping18.streamlit.app/ )  (athermal EOS mapping method, E-V mapping works and is incorrect for beta Sn while it is correct for Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization with incorrect output )

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer19-streamlit-red)](https://volume-expansion-athermal-eosmapping19.streamlit.app/ )  (athermal EOS mapping method,E-V mapping works and correct for both beta Sn and Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization with incorrect output)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer20-streamlit-red)](https://volume-expansion-athermal-eosmapping20.streamlit.app/ )  (athermal EOS mapping method,E-V mapping works and correct for both beta Sn and Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization is correct, phase stability calculation not yet complete)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer21-streamlit-red)](https://volume-expansion-athermal-eosmapping21.streamlit.app/ )  (athermal EOS mapping method,E-V mapping works and correct for both beta Sn and Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization is correct, phase stability calculation  complete)

[![atomisticmechanics](https://img.shields.io/badge/deltavolumer22-streamlit-red)](https://volume-expansion-athermal-eosmapping22.streamlit.app/ )  (athermal EOS mapping method,E-V mapping works and correct for both beta Sn and Li2Sn5,  plotly visualization works,  Energy minimization, Energy as a function of volume, mechanics and thermodynamics of phases and structures, DFT calculation with GPAW, fast and robust, all phases robust, structural visualization is correct, phase stability calculation  complete)








