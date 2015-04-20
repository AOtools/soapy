.. _dataSources:

Data Sources
============

In this section, the data sources which are stored in PyAOS are listed and a description of how they are obtained is given.


Simulation Run Data
-------------------
The following sources of data are recorded for each simulation run and are saved in a time stamped run specific directory inside the ``filePrefix`` directory. They can be acces by ``sim.<data>``, where ``<data>`` is listed in the  "Internal data structure" column. As the storing of some of these data sources can increase  memory usage significantly, they are not all saved by default, and the flag must be set in the configuration file.

+-------------+------------------+------------------+--------------------+
|Data         | Saved filename   |Internal data     |Comments            |
|             |                  |structure         |                    |
+=============+==================+==================+====================+
|Instantaneous|``instStrehl``    |``instStrehl``    |The Instantaneous   |
|Strehl ratio |                  |                  |Strehl ratio for    |
|             |                  |                  |each science target |
|             |                  |                  |frame               |
+-------------+------------------+------------------+--------------------+
|Long exposure|``longStrehl``    |``longStrehl``    |The Long exposure   |
|Strehl ratio |                  |                  |Strehl ratio for    |
|             |                  |                  |each science target |
|             |                  |                  |frame               |
+-------------+------------------+------------------+--------------------+
