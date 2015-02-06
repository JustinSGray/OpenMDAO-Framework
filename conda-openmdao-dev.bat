@ECHO OFF
REM cd to directory of conda build script
cd /d %~dp0
SET OLD_DIR=%CD%

REM Create an openmdao environment and
REM install numpy and scipy
REM TODO: Should move all possible openmdao dependencies up here to simplify script
SET DEPENDENCIES=pip
SET DEPENDENCIES=numpy
SET DEPENDENCIES=%DEPENDENCIES% scipy
SET DEPENDENCIES=%DEPENDENCIES% setuptools
SET DEPENDENCIES=%DEPENDENCIES% pyparsing
SET DEPENDENCIES=%DEPENDENCIES% traits==4.3.0
SET DEPENDENCIES=%DEPENDENCIES% nose
SET DEPENDENCIES=%DEPENDENCIES% sphinx==1.2.2
SET DEPENDENCIES=%DEPENDENCIES% "fabric==0.9.3"
SET DEPENDENCIES=%DEPENDENCIES% virtualenv
SET DEPENDENCIES=%DEPENDENCIES% boto
SET DEPENDENCIES=%DEPENDENCIES% paramiko==1.7.7.1
SET DEPENDENCIES=%DEPENDENCIES% requests
SET DEPENDENCIES=%DEPENDENCIES% decorator
SET DEPENDENCIES=%DEPENDENCIES% mock
SET DEPENDENCIES=%DEPENDENCIES% networkx
SET DEPENDENCIES=%DEPENDENCIES% zope.interface
SET DEPENDENCIES=%DEPENDENCIES% "pytz>=2011"
SET DEPENDENCIES=%DEPENDENCIES% pycrypto==2.3
SET DEPENDENCIES=%DEPENDENCIES% cobyla
SET DEPENDENCIES=%DEPENDENCIES% conmin
SET DEPENDENCIES=%DEPENDENCIES% newsumt
SET DEPENDENCIES=%DEPENDENCIES% slsqp
SET DEPENDENCIES=%DEPENDENCIES% bson
SET DEPENDENCIES=%DEPENDENCIES% pyevolve

@ECHO ON

conda create --yes --name openmdao %DEPENDENCIES%

@ECHO OFF
SET DEPENDENCIES=

REM Get the root directory of anaconda
cmd /c "activate openmdao && python -c "import sys; print sys.executable" > %TEMP%\python && deactivate"
SET /p PYTHON=<%TEMP%\python
DEL %TEMP%\python

REM install openmdao packages
cd openmdao.units
%PYTHON% setup.py develop
cd ..\openmdao.util
%PYTHON% setup.py develop
cd ..\openmdao.test
%PYTHON% setup.py develop
cd ..\openmdao.devtools
%PYTHON% setup.py develop
cd ..\openmdao.main
%PYTHON% setup.py develop
cd ..\openmdao.lib
%PYTHON% setup.py develop
REM install openmdao examples
cd ..\examples
cd openmdao.examples.bar3simulation
%PYTHON% setup.py develop
cd ..\openmdao.examples.expected_improvement
%PYTHON% setup.py develop
cd ..\openmdao.examples.mdao
%PYTHON% setup.py develop
cd ..\openmdao.examples.metamodel_tutorial
%PYTHON% setup.py develop
cd ..\openmdao.examples.nozzle_geometry_doe
%PYTHON% setup.py develop
cd ..\openmdao.examples.simple
%PYTHON% setup.py develop

SET PYTHON=
CD %OLD_DIR%
SET OLD_DIR=*/
@ECHO ON