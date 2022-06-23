seastar package
===============

**Top level package containing all code for the project.**
   
  
Subpackages
-----------

.. toctree::
   :maxdepth: 2

   seastar.gmfs
   seastar.examples
   seastar.utils
   seastar.retrieval



seastar.master_processor module
--------------------------------

**This module provides control for processing and can be run from command line.**
**Requires a local configuration file when instantiated. See below.**

.. sourcecode:: python

    obj = SEASTARX('seastarx_config.ini')
    obj.run()


.. automodule:: seastar.master_processor
   :members:
   :undoc-members:
   :show-inheritance:
