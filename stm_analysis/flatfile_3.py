#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""

    \mainpage Python Matrix Flat File Parser

    This script parse the Omicron Matrix Flat File Format to return a python
    object containing the measured data in physical units and all the relevant
    measurement parameters.

    \section Infos

    flatFile.py written by François Bianco, University of Geneva - francois.bianco@unige.ch


    \section Copyright

    Copyright © 2009-2011 François Bianco, University of Geneva - francois.bianco@unige.ch

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


    \section Updates
    2017-04 pconstantinou:
        - Added axis key for the latest MATRIX version; 'MATRIX V3.3.1-_v1639-tag-release-MatrixKit_3.3.1-195903'.
        - Explicitly defined 'sizeV' to be an integer for indexing purposes in I(V) spec and grid I(V).
    2016-12 tgill:
        Altered to be compatible with python 3.
    2016-04 tgill:
        Amended the info() object to also give information on user comments.
    2013-01 fbianco:
        added support for Matrix 3.1-1 which breaks the axis keys naming
    2011-11 fbianco:
        restructuring code
    2011-08-05 fbianco:
        code cleaning
    2009-05-06 fbianco:
        converted to a module
    2009-03-17 fbianco:
        added drawing capabilities
    2008-11-21 fbianco:
        first version

"""

from __future__ import division
from struct import unpack
import datetime
from pylab import *
import os.path
import numpy as np

DEBUG = False


class Error(Exception):
    """Base class for exceptions in this module. """
    pass


class UnhandledFileError(Error):
    """Occurs if the file has an unknown structure."""
    pass


class ParameterTypeError(Error):
    """Occurs if a parameter has an unknown type."""
    pass


class UnhandledDataType(Error):
    """Occurs if the file has an unknown data structure."""
    pass


class UnhandledTransferFunction(Error):
    """Occurs if the transfer function is unknown."""
    pass


class OutOfBoundError(Error):
    """Occurs if the value chosen is out of measured boundaries."""
    pass


class DataArray:
    """A simple class holding the minimal structure for storing STM data.
       The data is a numpy array with the right shape according to the type of
       data (i.e a vector for curve, a matrix for images, a 3d matrix for maps.)
       Info is a python dictionary to store physical information on the data.
    """
    def __init__(self, data, info):
        self.data = data # is a numpy matrix
        self.info = info.copy() # a simple dictionary


class FlatFile():
    """ The FlatFile class is able to parse the
        Omicron Flat File Format.
    """

    def __init__(self, filename):
        """ \arg filename should be a valid path to a omicron flat file."""

        self.filename = filename
        self.data = []  # List containing data of DataArray type

        # Define the keys in dictionary since they are version dependant
        # since the Matrix v3.1
        #
        # FIXME complete the list and correct the rest of the file parser
        #
        # Note we first use a dictionary of dictionary, then we override it
        # by the correct sub-dictionary, this was the faster to make the code
        # compatible with the breaking change of Matrix v3.1.
        #
        self.axis_keys = {}
        # MATRIX V3.3.1 - added by PCC
        self.axis_keys['MATRIX V3.3.1-_v1639-tag-release-MatrixKit_3.3.1-195903'] = {
            'V': 'Default::Spectroscopy::V',
            'X': 'Default::XYScanner::X',
            'Y': 'Default::XYScanner::Y',
            }
        self.axis_keys['MATRIX V3.1-1'] = {
            'V': 'Default::Spectroscopy::V',
            'X': 'Default::XYScanner::X',
            'Y': 'Default::XYScanner::Y',
            }
        self.axis_keys['MATRIX V3.0'] = {
            'V': 'V',
            'X': 'X',
            'Y': 'Y',
            }
        # Use V3.0 as fall-back value since it worked out up to V3.1
        self.axis_keys['fall-back'] = self.axis_keys['MATRIX V3.0']

        self.openFlatFile()

    def openFlatFile(self):
        """ Parse flatFile and create the data array with physical meaning
            based on the file structure.
        """

        # Open and read with binary flag
        self.file = open(os.path.normpath(self.filename), 'rb')

        #
        # Check Magic word and version
        #

        # Looking for magic word.
        self.magic_word = unpack('<4s', self.file.read(4))[0]
        if b'FLAT' != self.magic_word:
            raise UnhandledFileError('Magic word: %s is not FLAT'.format(self.magic_word))
        # Looking for file version.
        self.version = unpack('<4s', self.file.read(4))[0]
        if b'0100' != self.version:
            raise UnhandledDataType('Vernissage version: %s is not 0100'.format(self.version))

        #
        # Axis Hierarchy Description :
        #
        self.axis = {}

        # Number of axis
        axisCount = self._readInt()

        for i in range(axisCount) :

            # Axis name
            axisName = self._readString()

            self.axis[axisName] = {}

            # Trigger axis name
            self.axis[axisName]['trigger'] = self._readString()

            # Axis unit
            self.axis[axisName]['unit'] =  self._readString()

            # Clock count (number of points)
            self.axis[axisName]['clockCount'] = self._readInt()

            # Axis start value
            self.axis[axisName]['startValue'] = self._readInt()

            # Axis increment
            self.axis[axisName]['increment'] = self._readInt()

            # Axis start value physical
            self.axis[axisName]['startValuePhysical'] = self._readDouble()

            # Axis increment physical
            self.axis[axisName]['incrementPhysical'] = self._readDouble()

            # Mirrored
            self.axis[axisName]['mirrored'] = bool(self._readInt())

            # Table sets
            tableSetCount = self._readInt()

            self.axis[axisName]['tableSets'] = {}

            if DEBUG : print('Axis %i has %i table set(s)'.format(i, tableSetCount))

            for j in range(tableSetCount) :

                triggerAxisName = self._readString()
                self.axis[axisName]['tableSets'][triggerAxisName] = []

                intervalCount = self._readInt()

                if DEBUG : print('Trigger axis %s has %i intervals.'.format(triggerAxisName, intervalCount))

                for k in range(intervalCount) :
                    self.axis[axisName]['tableSets'][triggerAxisName].append({})

                    self.axis[axisName]['tableSets'][triggerAxisName][k]['start'] = self._readInt()

                    self.axis[axisName]['tableSets'][triggerAxisName][k]['stop'] = self._readInt()

                    self.axis[axisName]['tableSets'][triggerAxisName][k]['step'] = self._readInt()

        self.dimension = len(self.axis)


        #
        # Channel Description :
        #

        self.channel = {}

        # Channel name
        self.channel['name'] = self._readString()

        # Transfer fuction name
        #
        #     Known functions are :
        #         TFF_Linear1D : phys = ( raw - offset ) / f
        #         TFF_MultiLinear1D : phys = ( raw_1 - offset_pre ) * ( raw - offset ) / f_neutral / f_pre
        #
        transferFunctionName = self._readString()

        # Channel unit
        self.channel['unit'] = self._readString()

        # Transfer functions parameters
        parameterCount = self._readInt()
        parameters = {}

        for i in range(parameterCount) :
            # Parameter Name and value
            paramerterName = self._readString()
            parameters[paramerterName] = self._readDouble()

        if 'TFF_Linear1D' == transferFunctionName :
            transferFunction = lambda z: ( z - parameters['Offset'] ) / parameters['Factor']
        elif 'TFF_MultiLinear1D' == transferFunctionName :
            transferFunction = lambda z: ( parameters['Raw_1'] - parameters['PreOffset'] ) * ( z - parameters['Offset'] ) / parameters['NeutralFactor'] / parameters['PreFactor']
        else :
            raise UnhandledTransferFunction('File transfer function: %s is unknown.'.format(transferFunctionName))

        # Number of data views
        # -> Possible data view types :
        #
        #    1 : (2 Dim) vtc_Simple2D (unused)
        #    2 : (1 Dim) vtc_Simple1D
        #    3 : (2 Dim) vtc_ForwardBackward2D = Topography
        #    4 : (3 Dim) vtc_2Dof3D = Planes from a volume CITS data cube
        #    5 : (1/3 Dim) vtc_Spectroscopy
        #    6 : (1 Dim) vtc_ForceCurve
        #
        dataViewCount = self._readInt()
        self.dataView = []

        for j in range(dataViewCount) :
            self.dataView.append( self._readInt() )

        #
        # Creation information :
        #
        self.creationInformation = {}

        self.creationInformation['timestamp'] = unpack( '<q', self.file.read(8) )[0]
        self.creationInformation['date'] = datetime.datetime.fromtimestamp( float(self.creationInformation['timestamp']) ).isoformat(' ')
        self.creationInformation['comment'] = self._readString() ## Added by TGG


        #
        # Raw Data :
        #

        # Total number of data elements excepted
        self.brickletSize = self._readInt()

        # Actual number of data elements measured
        self.dataItemSize = self._readInt()

        # Raw data array
        self.rawData = []

        for i in range(self.dataItemSize) :
            self.rawData.append( transferFunction(self._readInt()) )
        # The void pixels will be automatically filled with 0
        # when using array.resize() with a bigger size than its actual size
        # This is done in self.reshapeData()

        #
        # Sample position information
        #
        offsetCount = self._readInt()

        self.offset = []

        for i in range(offsetCount) :
            self.offset.append( (self._readDouble(),self._readDouble()) )

        #
        # Experiment information
        #
        self.experimentInfo = {}
        self.experimentInfo['Name'] = self._readString()
        self.experimentInfo['Version'] = self._readString()
        self.experimentInfo['Description'] = self._readString()
        self.experimentInfo['File Specification'] = self._readString()
        self.experimentInfo['File Creator'] = self._readString()
        self.experimentInfo['Result File Creator'] = self._readString()
        self.experimentInfo['User Name'] = self._readString()
        self.experimentInfo['Account Name'] = self._readString()
        self.experimentInfo['Result Data File Specification'] = self._readString()
        self.experimentInfo['Run Cycle'] = self._readInt()
        self.experimentInfo['Scan Cycle'] = self._readInt()

        #
        # Select axis keys from the Matrix version
        #
        if DEBUG:
            print('File creator', self.experimentInfo['Result File Creator'])
        if self.experimentInfo['Result File Creator'] in self.axis_keys.keys():
            self.axis_keys = self.axis_keys[
                self.experimentInfo['Result File Creator']]
        else:
            print('WARNING: Missing axis key for %s,'.format(self.experimentInfo['Result File Creator']))
            print('trying fall-back values.')
            print('')
            print('If the data file does not load, add a new axis keys')
            print('dictionary in the source code.')
            self.axis_keys = self.axis_keys['fall-back']

        #
        # Experiment Element Parameter List
        #
        elementsCount = self._readInt()

        self.experimentElement = {}

        for i in range(elementsCount) :

            instanceName = self._readString()
            self.experimentElement[instanceName] = {}

            parameterCount = self._readInt()

            for j in range(parameterCount) :

                parameterName = self._readString()
                parameterTypeCode = self._readInt()
                parameterUnit = self._readString()
                parameterValue = self._readString()

                # Every value is passed as a string but can
                # represent different object type according
                # to the given parameter type code
                if parameterTypeCode == 1 :   # 32bits integer
                    parameterValue = int(parameterValue)
                elif parameterTypeCode == 2 : # Double precision float
                    parameterValue = float(parameterValue)
                elif parameterTypeCode == 3 : # Boolean
                    if parameterValue == 'true' :
                        parameterValue = True
                    else :
                        parameterValue = False
                elif parameterTypeCode == 4 : # Enum
                    parameterValue = parameterValue # FIXME unhandled, since never used
                elif parameterTypeCode == 5 : # Unicode character string
                    parameterValue = parameterValue
                else :
                    raise ParameterTypeError('Unknown parameter type %i given'.format(parameterTypeCode))

                self.experimentElement[instanceName][parameterName] = {
                    'value': parameterValue,
                    'unit' : parameterUnit
                    }
                # Note: parameterUnit can be = u'--' which means no unit.

        #
        # Deployement parameters
        #
        elementsCount = self._readInt()

        self.experimentDeployement = {}

        for i in range(elementsCount) :

            instanceName = self._readString()
            deploymentCount = self._readInt()

            self.experimentDeployement[instanceName] = {}

            for j in range(deploymentCount) :

                 self.experimentDeployement[instanceName][self._readString()] = self._readString()

        assert self.file.read() == b'', 'There are still some unknown information at the end of the file %s '.format(self.filename)

        self.file.close()
        self.file = None # Explicitly delete the file object

        # Deal with the real stuff, try to reconstruct the real data shape from
        # the raw data.
        self._reshapeData()


    def _readString( self ) :
        """Read a Omicron string in the open file. The string are stored as UTF-16 characters preceded with an integer corresponding to the length of the string """

        length = unpack( '<i', self.file.read(4) )[0]
        if length:
            return str(self.file.read(2 * length), encoding='utf16', errors='replace') # 16 bits unicode character
        else:
            return None


    def _readInt( self ) :
        """Unpack an integer from the bytestream """

        return unpack( '<i', self.file.read(4) )[0]


    def _readDouble( self ) :
        """Unpack a double from the bytestream """

        return unpack( '<d', self.file.read(8) )[0]

    def _reshapeData(self):
        """Create a data dictionary from the rawData according to the file parameters """

        self.rawData = np.array(self.rawData) # Convert data list to a numpy array

        # common info for all type of files
        info = {'filename' : self.filename,
                'comment': self.creationInformation['comment'],  # Added by TGG.
                'unit' : self.channel['unit'],
                'date' : self.creationInformation['date'],
                'runcycle' : 'Run %i – cycle %i\n' % \
                    (self.experimentInfo['Run Cycle'],
                     self.experimentInfo['Scan Cycle']),
                'current' : self.experimentElement['Regulator']\
                                        ['Setpoint_1']['value'],
                'vgap' : self.experimentElement['GapVoltageControl']\
                                        ['Voltage']['value'],
                'offset' : self.offset
                }

        if self.isTopography():

            sizeX = int(self.axis[self.axis_keys['X']]['clockCount']/(self.axis[self.axis_keys['X']]['mirrored']+1))
            sizeY = int(self.axis[self.axis_keys['Y']]['clockCount']/(self.axis[self.axis_keys['Y']]['mirrored']+1))

            info.update({
                'type' : 'topo',
                'xres' : sizeX,
                'yres' : sizeY,
                'xinc' : self.axis[self.axis_keys['X']]['incrementPhysical'] * 1e9,
                'yinc' : self.axis[self.axis_keys['Y']]['incrementPhysical'] * 1e9,
                'xreal' : self.axis[self.axis_keys['X']]['incrementPhysical'] * sizeX * 1e9,
                'yreal' : self.axis[self.axis_keys['Y']]['incrementPhysical'] * sizeY * 1e9,
                'unitxy' : 'nm',
                })

            self.rawData.resize( sizeY*(self.axis[self.axis_keys['Y']]['mirrored']+1), sizeX*(self.axis[self.axis_keys['X']]['mirrored']+1) )

            # Both axis are mirrored
            # 4 images : up-fwd, up-bwd, down-fwd, down-bwd
            # Note on array syntax [start:stop:increment]
            if self.axis[self.axis_keys['X']]['mirrored'] and self.axis[self.axis_keys['Y']]['mirrored'] :
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(self.rawData[ 0:sizeY, 0:sizeX], info))
                info['direction'] = 'up-bwd'
                self.data.append(DataArray(self.rawData[ 0:sizeY, :sizeX-1:-1 ], info))
                info['direction'] = 'down-fwd'
                self.data.append(DataArray(self.rawData[ :sizeY-1:-1, 0:sizeX], info))
                info['direction'] = 'down-bwd'
                self.data.append(DataArray( self.rawData[ :sizeY-1:-1, :sizeX-1:-1], info))

            # Only X is mirrored
            # 2 images up : fwd and bwd
            elif self.axis[self.axis_keys['X']]['mirrored'] :
                if DEBUG : print('Only X is mirrored')
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(self.rawData[ :,0:sizeX], info))
                info['direction'] = 'up-bwd'
                self.data.append(DataArray(self.rawData[ :,:sizeX-1:-1], info))

            # Only Y is mirrored
            # 2 images fwd : up and down
            elif self.axis[self.axis_keys['Y']]['mirrored'] :
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(self.rawData[ 0:sizeY,:], info))
                info['direction'] = 'down-fwd'
                self.data.append(DataArray(self.rawData[ :sizeY-1:-1,:], info))

            # Only one image
            else :
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(self.rawData, info))

        elif self.isVPointSpectroscopy():
            # PCC changed sizeV to be explicitely an integer only for indexing purposes
            sizeV = int(self.axis[self.axis_keys['V']]['clockCount']/(self.axis[self.axis_keys['V']]['mirrored']+1))

            info.update({
                'type' : 'ivcurve',
                'vres' : sizeV,
                'vstart' : self.axis[self.axis_keys['V']]['startValuePhysical'],
                'vinc' : self.axis[self.axis_keys['V']]['incrementPhysical'],
                'vreal' : sizeV * self.axis[self.axis_keys['V']]['incrementPhysical'],
                'unitv' : self.axis[self.axis_keys['V']]['unit'],
                })

            info['direction'] = 'fwd'
            self.data.append(DataArray(self.rawData[:sizeV], info))
            if self.axis[self.axis_keys['V']]['mirrored']:
                info['direction'] = 'bwd'
                self.data.append(DataArray(self.rawData[:sizeV-1:-1], info))

        elif self.isZPointSpectroscopy():
            # FIXME Implement izcurve
            print("Not implemented")

            #zScale = self.axis['Z']['startValuePhysical'] + arange(self.axis['Z']['clockCount'])* #self.axis['Z']['incrementPhysical']


        elif self.isGridSpectroscopy():

            infoX = self.axis[self.axis_keys['V']]['tableSets'][self.axis_keys['X']][0]
            infoY = self.axis[self.axis_keys['V']]['tableSets'][self.axis_keys['Y']][0]

            # this are already the sizes of the sub-images
            # i.e we do not need to divide them for mirrored images
            sizeX = (infoX['stop']-infoX['start'])//infoX['step']+1
            sizeY = (infoY['stop']-infoY['start'])//infoY['step']+1

            mirroredV = self.axis[self.axis_keys['V']]['mirrored']
            sizeV = int(self.axis[self.axis_keys['V']]['clockCount']/(mirroredV+1))

            # Find out if I(V) are measured on bwd and fwd scan (==mirrored)
            mirroredX = len(self.axis[self.axis_keys['V']]['tableSets'][self.axis_keys['X']])==2
            mirroredY = len(self.axis[self.axis_keys['V']]['tableSets'][self.axis_keys['Y']])==2

            # If I(V) are measured on bwd and fwd, the axis should be mirrored
            if ( mirroredX and not self.axis[self.axis_keys['X']]['mirrored'] ) or ( mirroredY and not self.axis[self.axis_keys['Y']]['mirrored']):
                raise UnhandledDataType("The file %s has an unknown structure".format(self.filename))

            info.update({
                'type' : 'ivmap',
                'xres' : sizeX,
                'yres' : sizeY,
                'xinc' : self.axis[self.axis_keys['X']]['incrementPhysical'] * 1e9,
                'yinc' : self.axis[self.axis_keys['Y']]['incrementPhysical'] * 1e9,
                'xreal' : self.axis[self.axis_keys['X']]['incrementPhysical'] * sizeX * 1e9,
                'yreal' : self.axis[self.axis_keys['Y']]['incrementPhysical'] * sizeY * 1e9,
                'unitxy' : 'nm',
                'vres' : sizeV,
                'vstart' : self.axis[self.axis_keys['V']]['startValuePhysical'],
                'vinc' : self.axis[self.axis_keys['V']]['incrementPhysical'],
                'vreal' : sizeV * self.axis[self.axis_keys['V']]['incrementPhysical'],
                'unitv' : self.axis[self.axis_keys['V']]['unit'],
            })
            dataTemp = np.copy(self.rawData) # FIXME copy() .... this has huge memory impact
            dataTemp.resize(sizeX*(mirroredX+1)*sizeY*(mirroredY+1),
                            sizeV*(mirroredV+1)) # each line is a spect. curve
            dataTemp = np.transpose(dataTemp) # each column is a spectroscopy curve

            # Cut Matrix in two if data are mirrored
            if mirroredV:
                dataTempMirrored = np.copy(dataTemp[:sizeV-1:-1,:]) # Reverse order of mirrored data
                dataTemp = np.copy(dataTemp[:sizeV,:])

            dataTemp = np.resize(dataTemp, (sizeV,
                             sizeY*(mirroredY+1),
                             sizeX*(mirroredX+1))) # slices,cols,rows : 3D view
            if mirroredV:
                dataTempMirrored.resize(sizeV,
                                        sizeY*(mirroredY+1),
                                        sizeX*(mirroredX+1))     # slices,cols,rows : 3D view

            self.data = []

            # Both axis are mirrored
            # 4 images : up-fwd, up-bwd, down-fwd, down-bwd
            # Note on array syntax [start:stop:increment]
            if mirroredX and mirroredY :
                if DEBUG : print('X and Y mirrored')
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(dataTemp[ :, 0:sizeY, 0:sizeX], info))
                info['direction'] = 'up-bwd'
                self.data.append(DataArray(dataTemp[ :, 0:sizeY, :sizeX-1:-1 ], info))
                info['direction'] = 'down-fwd'
                self.data.append(DataArray(dataTemp[ :, :sizeY-1:-1, 0:sizeX], info))
                info['direction'] = 'down-bwd'
                self.data.append(DataArray(dataTemp[ :, :sizeY-1:-1, :sizeX-1:-1], info))

                if mirroredV:
                    if DEBUG : print('V mirrored')
                    info['direction'] = 'up-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, 0:sizeY, 0:sizeX], info))
                    info['direction'] = 'up-bwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, 0:sizeY, :sizeX-1:-1 ], info))
                    info['direction'] = 'down-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, :sizeY-1:-1, 0:sizeX], info))
                    info['direction'] = 'down-bwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, :sizeY-1:-1, :sizeX-1:-1], info))

            # Only X is mirrored
            # 2 images up : fwd and bwd
            elif mirroredX:
                if DEBUG : print('X mirrored only')
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(dataTemp[ :, :, 0:sizeX ], info))
                info['direction'] = 'up-bwd'
                self.data.append(DataArray(dataTemp[ :, :, :sizeX-1:-1 ], info))

                if mirroredV:
                    if DEBUG : print('V mirrored')
                    info['direction'] = 'up-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, :, 0:sizeX ], info))
                    info['direction'] = 'up-bwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, :, :sizeX-1:-1 ], info))

            # Only Y is mirrored
            # 2 images fwd : up and down
            elif mirroredY:
                if DEBUG : print('Y mirrored only')
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(dataTemp[ :, 0:sizeY,:], info))
                info['direction'] = 'down-fwd'
                self.data.append(DataArray(dataTemp[ :, :sizeY-1:-1,:], info))

                if mirroredV:
                    if DEBUG : print('V mirrored')
                    info['direction'] = 'up-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, 0:sizeY,:], info))
                    info['direction'] = 'down-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored[ :, :sizeY-1:-1,:], info))

            # Only one image
            else:
                if DEBUG : print('X, Y not mirrored')
                info['direction'] = 'up-fwd'
                self.data.append(DataArray(dataTemp, info))

                if mirroredV:
                    if DEBUG : print('V mirrored')
                    info['direction'] = 'up-fwd mirrored'
                    self.data.append(DataArray(dataTempMirrored, info))

        else :
            if DEBUG:
                print(self.axis)
            raise UnhandledDataType("The data file %s has an unhandled type.",format(self.filename))

    def isTopography(self):
        """ Return True if the file represents a topography image with X and Y axes. """
        if self.dimension == 2 and self.axis_keys['X'] in self.axis and self.axis_keys['Y'] in self.axis:
            return True
        else:
            return False

    def isVPointSpectroscopy(self):
        """ Return True if the file represents a point spectroscopy. """

        if self.dimension == 1 and self.axis_keys['V'] in self.axis:
            return True
        else:
            return False

    def isZPointSpectroscopy(self):
        """ Return True if the file represents a Z spectroscopy. """

        if self.dimension == 1 and self.axis_keys['Z'] in self.axis:
            return True
        else:
            return False

    def isGridSpectroscopy(self):
        """ Return True if the file represents a grid spectroscopy. """

        if self.dimension == 3 and \
           self.axis_keys['X'] in self.axis and \
           self.axis_keys['Y'] in self.axis and \
           self.axis_keys['V'] in self.axis:
            return True
        else:
            return False

    def getData(self):
        """Return the read data"""

        return self.data

def load(filename):
    """Loader function for further data processing
    Return a list of DataArray object"""

    ff = FlatFile(filename)
    return ff.getData()

if __name__ == "__main__":
    pass