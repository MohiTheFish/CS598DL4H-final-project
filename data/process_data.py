import sys
import os
import argparse
import pickle
from datetime import datetime
from collections import defaultdict

def convert_to_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
		else: return dxStr
	
def convert_to_3digit_icd9(dxStr):
	if dxStr.startswith('E'):
		if len(dxStr) > 4: return dxStr[:4]
		else: return dxStr
	else:
		if len(dxStr) > 3: return dxStr[:3]
		else: return dxStr

def getOrderedPatientsInfo(patientInfos):
	types = {}
	newPatientInfos = []
	for patient in patientInfos:
		newPatient = []
		for visit in patient:
			newVisit = []
			for code in set(visit):
				if code in types:
					newVisit.append(types[code])
				else:
					types[code] = len(types)
					newVisit.append(types[code])
			newPatient.append(newVisit)
		newPatientInfos.append(newPatient)
	return types, newPatientInfos


def parse_arguments(parser):
	"""Read user arguments"""
	parser.add_argument(
		"--mimic_dir", type=str, default='mimic-iii-clinical-database-1.4/', help="Directory for MIMIC-III data"
	)
	parser.add_argument(
		"--admission_file", type=str, default='ADMISSIONS.csv', help="ADMISSIONS data file"
	)
	parser.add_argument(
		"--diagnosis_file", type=str, default='DIAGNOSES_ICD.csv', help="DIAGNOSES data file"
	)
	parser.add_argument(
		"--patients_file", type=str, default='PATIENTS.csv', help="PATIENTS data file"
	)
	parser.add_argument(
		"--prescriptions_file", type=str, default='PRESCRIPTIONS.csv', help="PRESCRIPTIONS data file"
	)
	parser.add_argument(
		"--outdir", type=str, default="processed_data/", help="dir to output processed files to"
	)
	args = parser.parse_args()

	return args
if __name__ == '__main__':
	PARSER = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	ARGS = parse_arguments(PARSER)
	admissionFile = ARGS.mimic_dir + ARGS.admission_file
	diagnosisFile = ARGS.mimic_dir + ARGS.diagnosis_file
	patientsFile = ARGS.mimic_dir + ARGS.patients_file
	prescriptionFile = ARGS.mimic_dir + ARGS.prescriptions_file
	outDir = ARGS.outdir
	if not outDir.endswith('/'):
		outDir = outDir+'/'

	print('Collecting mortality information')
	pidDodMap = {}
	infd = open(patientsFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		pid = int(tokens[1])
		dod_hosp = tokens[5]
		if len(dod_hosp) > 0:
			pidDodMap[pid] = 1
		else:
			pidDodMap[pid] = 0
	infd.close()

	print('Building pid-admission mapping, admission-date mapping')
	pidAdmMap = defaultdict(list)
	admDateMap = {}
	infd = open(admissionFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		pid = int(tokens[1])
		admId = int(tokens[2])
		admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
		admDateMap[admId] = admTime
		pidAdmMap[pid].append(admId)
	infd.close()

	print('Building admission-dxList mapping')
	admDxMap = defaultdict(list)
	admDxMap_3digit = defaultdict(list)
	infd = open(diagnosisFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		admId = int(tokens[2])
		dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
		dxStr_3digit = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])

		admDxMap[admId].append(dxStr)
		admDxMap_3digit[admId].append(dxStr_3digit)
	infd.close()

	print('Building admission-prescription mapping')
	uniquePrescriptions = set()
	admPrescriptionMap = defaultdict(list)
	infd = open(prescriptionFile, 'r')
	infd.readline()
	for line in infd:
		tokens = line.strip().split(',')
		admId = int(tokens[2])
		prescriptionStr = tokens[10] # The Formulary_Drug_CD

		admPrescriptionMap[admId].append(prescriptionStr)
		uniquePrescriptions.add(prescriptionStr)
	numUniquePrescriptions = len(uniquePrescriptions)

	print('Building pid-sortedVisits mapping')
	pidSeqMap = {}
	pidSeqMap_3digit = {}
	for pid, admIdList in pidAdmMap.items():
		if len(admIdList) < 2: continue

		sortedList = sorted([(admDateMap[admId], admDxMap[admId], admPrescriptionMap[admId]) for admId in admIdList])
		pidSeqMap[pid] = sortedList

		sortedList_3digit = sorted([(admDateMap[admId], admDxMap_3digit[admId], admPrescriptionMap[admId]) for admId in admIdList])
		pidSeqMap_3digit[pid] = sortedList_3digit
	
	print('Building pids, dates, mortality_labels, strSeqs')
	pids = []
	dates = []
	seqs = []
	prescriptions = []
	morts = []
	for pid, visits in pidSeqMap.items():
		pids.append(pid)
		morts.append(pidDodMap[pid])
		seq = []
		date = []
		prescription = []
		for visit in visits:
			date.append(visit[0])
			seq.append(visit[1])
			prescription.append(visit[2])
		dates.append(date)
		seqs.append(seq)
		prescriptions.append(prescription)
	
	print('Building pids, dates, strSeqs for 3digit ICD9 code')
	seqs_3digit = []
	for pid, visits in pidSeqMap_3digit.items():
		seq = []
		for visit in visits:
			seq.append(visit[1])
		seqs_3digit.append(seq)
	
	print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
	types_3digit, newSeqs_3digit = getOrderedPatientsInfo(seqs_3digit)

	print('Converting prescriptions to something, and making types')
	types_prescriptions, newPrescriptions = getOrderedPatientsInfo(prescriptions)

	print()
	print(f'numUniquePrescriptions: {numUniquePrescriptions}')
	if not os.path.isdir(outDir):
		os.mkdir(outDir)
	
	pickle.dump(pids, open(outDir+'pids.pkl', 'wb'), -1)
	pickle.dump(dates, open(outDir+'dates.pkl', 'wb'), -1)
	pickle.dump(morts, open(outDir+'morts.pkl', 'wb'), -1)
	pickle.dump(newSeqs_3digit, open(outDir+'3digitICD9.seqs.pkl', 'wb'), -1)
	pickle.dump(types_3digit, open(outDir+'3digitICD9.types.pkl', 'wb'), -1)
	pickle.dump(newPrescriptions, open(outDir+'prescriptions.pkl', 'wb'), -1)
	pickle.dump(types_prescriptions, open(outDir+'prescriptions.types.pkl', 'wb'), -1)
