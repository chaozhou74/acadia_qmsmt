# Example data folders

> **Note:** the absolute paths below point at one specific station's archive and will
> not resolve elsewhere — this file is a research record of a breadth test, kept for the
> class-to-behaviour mapping (what each runtime exercises), not as portable examples.

Every distinct runtime class under `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full`
(7875 archived runs, 39 classes) was traced as a
breadth test. **37 of 39 trace cleanly** and `compare_with_compiled_log` matches on every
one of them; the remaining 2 are blocked on purpose because they drive external
instruments (see `DEVELOPER_NOTES.md`, *Instrument safety*).

Columns: `pts` sweep points captured, `blk` compiled blocks, `plc` executed placements
(loops unrolled, untaken `test` bodies dropped), `dead` inter-block dead time,
`cf` control flow present, `reg` register-driven lengths.

## Start with these

| folder | class | why it is interesting |
|---|---|---|
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_rabi/260727_010800` | QubitPulseAmplitudeCalibrationRuntime | the simplest thing there is: one block, no dead time, no control flow |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_t2e/260727_010933` | QubitCoherenceRuntime | register-driven delay — the timeline *length* changes as you step points |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_R1_chevron/260727_011131` | BSChevronRuntime | 4096 sweep points off a single compiled schedule |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/1_Qubit_Tomo/260723/164303` | OneQubitTomographyRuntime | 12 compiled blocks, 2 executed: `test` picks the tomography axis per point |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/AllXY_prelim_test/260723/165848` | AllXYRuntime | 42 compiled blocks, 2 executed — the widest compiled-vs-executed gap here |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/R1_coherence/260727_011426` | InterleavedCoherenceMsmtSingleQMRuntime | **the showcase**: `repeat_until` reset loops *and* a `test`, 3 registers, 24→13 blocks |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/Bus_coherence_qccq/260727_011943` | InterleavedCoherenceMsmtQCCQRuntime | same family, 15→4 blocks |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/Qubit_RB/260723/165326` | QubitRBRuntime | the Clifford comes off the bus at runtime — a command with no length at all |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_BeamSplitting/LA_BeamSplitting/QM_QM_BS_AmpSweep_DR_Tomo/LA_L1_Bus/260726/173818` | QmQmBSSweepAmpDRTomoRuntime | 30 µs, memories reloaded per point (`swap_copy1`) — step points to see them change |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RB_Calibration/qubit_cal/bs_blob_cal/20260710_134841/swap_swapRep064` | CavityCavityBSAmpNcoSweepBlobsRuntime | 17 blocks, 41 µs — the longest sequence in the archive |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/THE_PROTOCOL/260720/232742` | THEPROTOCOLRuntime | 13 blocks and 1175 ns of dead time — the most boundary-dominated one |
| `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_BeamSplitting/1DR_Tomo/L2_L3/260721/235225` | OneDRTomographyRuntime | 6 blocks, 3 points, dual-rail tomography |

## Every class

| class | runs | pts | blk | plc | len µs | dead ns | cf | reg | folder |
|---|--:|--:|--:|--:|--:|--:|---|---|---|
| AllXYRuntime | 6 | 105 | 42 | 2 | 1.96 | 100 | test | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/AllXY_prelim_test/260723/165848` |
| BSAmpNcoSweepPulseNcoBlobsRuntime | 176 | 3362 | 15 | 15 | 23.74 | 65 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RB_Calibration/qubit_cal/bs_blob_cal/20260713_133605/swap_swapRep064` |
| BSChevronQSwitchRuntime | 2 | 4096 | 2 | 2 | 1.82 | 100 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_BeamSplitting/BS_Chevron_QSwitch/LB_qb_L1/260715/181952` |
| BSChevronRuntime | 315 | 4096 | 2 | 2 | 1.38 | 100 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_R1_chevron/260727_011131` |
| BellPairCavitiesViaBus | 1 | 36 | 3 | 3 | 3.40 | 235 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/Bell_Pair_Cavities_Simultaneous_BS/260723/185758` |
| BerryChevronRuntime | 38 | 4096 | 3 | 3 | 2.19 | 175 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_LB_Calibration/qubit_cal/qb_l2_berry_chevron/260715_155335` |
| BerryDRAmpFreqSweepRuntime | 30 | 4096 | 11 | 11 | 9.20 | 515 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_LB_Calibration/qubit_cal/l2_l3_cz_cal/260715_164339` |
| BerryGateTomographyRuntime | 7 | 36 | 4 | 4 | 3.51 | 275 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RB_BeamSplitting/Berry_Qubit_Cav_Tomo/RB_qb_R2/260723/175522` |
| CavityCavityBSAmpFreqSweepBlobsRuntime | 2 | 3362 | 7 | 7 | 16.18 | 225 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RB_Calibration/qubit_cal/qb_r2_r3_blob_cal_swap/260708_162157` |
| CavityCavityBSAmpNcoSweepBlobsRuntime | 190 | 3362 | 17 | 17 | 41.08 | 225 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RB_Calibration/qubit_cal/bs_blob_cal/20260710_134841/swap_swapRep064` |
| CavityCavityChevronRuntime | 443 | 4096 | 5 | 5 | 2.39 | 290 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_LB_Calibration/bs_amp_sweep/L1_Bus4/20260727_122102/bs_amp_0.35000/L1_Bus_chevron_no_update/260727_123543` |
| InterleavedCoherenceMsmtQCCQRuntime | 64 | 303 | 15 | 4 | 2.44 | 265 | test | REG0,REG1,REG2 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/Bus_coherence_qccq/260727_011943` |
| InterleavedCoherenceMsmtSingleQMRuntime | 145 | 383 | 24 | 13 | 7.73 | 1095 | repeat_until,test | REG0,REG1,REG2 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/R1_coherence/260727_011426` |
| OneCavityTomographyRuntime | 29 | 6 | 18 | 3 | 2.04 | 165 | test | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RB_BeamSplitting/1_Cav_Tomo/RB_qb_R1/260720/235931` |
| OneDRTomographyRuntime | 79 | 3 | 6 | 6 | 7.24 | 390 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_BeamSplitting/1DR_Tomo/L2_L3/260721/235225` |
| OneQubitTomographyRuntime | 10 | 6 | 12 | 2 | 1.97 | 105 | test | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/1_Qubit_Tomo/260723/164303` |
| QMT1Runtime | 5 | 51 | 4 | 4 | 2.25 | 230 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_QM_Char/QM_T1/LB_qb_L1/260721/175252` |
| QMT1TwoSWAPSRuntime | 17 | 51 | 4 | 4 | 3.60 | 235 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RB_Calibration/qubit_cal/R3_T1/260707_175048` |
| QMT2Runtime | 13 | 51 | 6 | 6 | 2.87 | 370 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_QM_Char/QM_T2E/LB_qb_L1/260707/152956` |
| QMT2TwoSWAPSRuntime | 8 | 51 | 5 | 5 | 3.28 | 310 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RB_QM_Char/QM_T2_twoSWAPs/R1_Bus/260723/231039` |
| QmQmBSSweepAmpDRTomoRuntime | 17 | 96 | 5 | 5 | 29.82 | 360 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_BeamSplitting/LA_BeamSplitting/QM_QM_BS_AmpSweep_DR_Tomo/LA_L1_Bus/260726/173818` |
| QubitAnharmonicityRuntime | 44 | 101 | 1 | 1 | 7.08 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RB_Qubit_Char/Qubit_efSpec/260629/190939` |
| QubitCoherenceRuntime | 152 | 151 | 5 | 5 | 2.08 | 305 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_t2e/260727_010933` |
| QubitEFPulseAmplitudeCalibrationRuntime | 30 | 101 | 1 | 1 | 1.85 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/Qubit_efRabi/260715/182951` |
| QubitPulseAmplitudeCalibrationRuntime | 137 | 101 | 1 | 1 | 1.24 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_rabi/260727_010800` |
| QubitQmBSSweepAmpDRTomoRuntime | 8 | 78 | 4 | 4 | 8.09 | 290 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RA_BeamSplitting/QM_BS_DR_Tomo/RA_qb_R1/260724/162702` |
| QubitRBRuntime | 9 | 61 | 4 | 4 | 4.43 | 235 | repeat_until,test | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/Qubit_RB/260723/165326` |
| QubitRelaxationRuntime | 50 | 51 | 2 | 2 | 1.44 | 100 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_t1/260727_010831` |
| QubitSpectroscopyRuntime | 177 | 151 | 1 | 1 | 5.21 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/qb_spec/260727_010731` |
| ReadoutBenchmarkingRuntime | 3 | 1 | 2 | 2 | 2.37 | 105 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RA_Qubit_Char/Readout_Benchmarking/260724/174618` |
| ReadoutFidelityRuntime | 2444 | 2 | 2 | 1 | 1.82 | 0 | test | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/RA_RO_Calibration/readout_scale_0.900_readout_flat_1.000_readout_frequency_9035.000/Readout_Hist_and_Metrics/260720_073629` |
| ReadoutWindowCalibrationRuntime | 2568 | 2 | 1 | 1 | 1.24 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/ro_window/260727_010824` |
| ResonatorSpectroscopyFluxSweepRuntime | 24 | — | — | — | — | — | **instruments: blocked** | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LA_SPA_Spec_FluxSweep/260630/234849` |
| ResonatorSpectroscopyPrepQubitRuntime | 22 | 402 | 1 | 1 | 1.69 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LB_Qubit_Char/Readout_Spec_prep_qubit/260630/215855` |
| ResonatorSpectroscopyRuntime | 152 | 101 | 1 | 1 | 1.02 | 0 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/FT_RA_Calibration/qubit_cal/ro_spec/260727_010725` |
| ResonatorSpectroscopySpaOnOffRuntime | 133 | — | — | — | — | — | **instruments: blocked** | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/LA_Qubit_Char/Readout_Spec/SPA_on_vs_off/260630/233941` |
| SimultaneousDriveAmpNcoSweepBlobsRuntime | 96 | 3362 | 6 | 6 | 7.65 | 295 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/BS_SimulBus/Blobs_NCO/260710/232405` |
| SimultaneousDriveChevronRuntime | 64 | 4096 | 5 | 5 | 2.79 | 360 | — | REG0 | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/BS_SimulBus/Chevron/260723/233759` |
| THEPROTOCOLRuntime | 165 | 36 | 13 | 13 | 10.31 | 1175 | — | — | `/mnt/Data/FlexTanglement/20260626_Neo/FT_Full/THE_PROTOCOL/260720/232742` |

## Notes

- `blk` > `plc` means control flow: a `test` whose body was resolved away. AllXY is the
  extreme — 42 compiled blocks, 2 executed, because the sequence compiles every one of
  the 21 pulse pairs and picks one at runtime.
- `plc` > `blk` would mean an unrolled loop. No archived runtime here uses `loop(N)`;
  that path is validated by `validation/loopback_timing_cases.py` (`loop_2`, `loop_3`,
  `loop_2_double`).
- The two `repeat_until` users are the interleaved-coherence runtimes — both are active
  reset, so the real sequence is *longer* than the one pass drawn.
- `pts` is what one dry run captured. Every point shares the compiled schedule, so
  `trace.select_point(i)` switches instantly with no re-trace.
- Regenerate this table after changing the tracer; a class that stops tracing is a
  regression the selftest will not catch.
