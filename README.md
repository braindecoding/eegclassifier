# eegclassifier
BrainDigiCNN based

## Dataset Info
EP1.01.txt adalah dataset dari device emotiv epoc(EP).

FILE FORMAT:

The data is stored in a very simple text format including:

[id]: a numeric, only for reference purposes.

[event] id, a integer, used to distinguish the same event captured at different brain locations, used only by multichannel devices (all except MW).

[device]: a 2 character string, to identify the device used to capture the signals, "MW" for MindWave, "EP" for Emotive Epoc, "MU" for Interaxon Muse & "IN" for Emotiv Insight.

[channel]: a string, to indentify the 10/20 brain location of the signal, with possible values:
 
MindWave	"FP1"
EPOC	"AF3, "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
Muse	"TP9,"FP1","FP2", "TP10"
Insight	"AF3,"AF4","T7","T8","PZ" 

[code]: a integer, to indentify the digit been thought/seen, with possible values 0,1,2,3,4,5,6,7,8,9 or -1 for random captured signals not related to any of the digits.

[size]: a integer, to identify the size in number of values captured in the 2 seconds of this signal, since the Hz of each device varies, in "theory" the value is close to 512Hz for MW, 128Hz for EP, 220Hz for MU & 128Hz for IN, for each of the 2 seconds.

[data]: a coma separated set of numbers, with the time-series amplitude of the signal, each device uses a different precision to identify the electrical potential captured from the brain: integers in the case of MW & MU or real numbers in the case of EP & IN.

There is no headers in the files,  every line is  a signal, and the fields are separated by a tab

For example one line of each device could be (without the headers)

[id]	[event]	[device]	[channel]	[code]	[size]	[data]
27	27	MW	FP1	5	952	18,12,13,12,5,3,11,23,37,36,26,24,35,42â€¦â€¦
67650	67636	EP	F7	7	260	4482.564102,4477.435897,4484.102564â€¦â€¦.
978210	132693	MU	TP10	1	476	506,508,509,501,497,494,497,490,490,493â€¦â€¦
1142043	173652	IN	AF3	0	256	4259.487179,4237.948717,4247.179487,4242.051282â€¦â€¦
