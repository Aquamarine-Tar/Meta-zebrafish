#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include <deque>
#include <string>
#include "fem/World.h"
#include "Worm.h"
#include "KeyWorm.h"
#include "Normalizer.h"
#include "util/ArgParser.h"
#include "fluid/PeridynoBridge.h"

class Worm;
class Normalizer;
class Environment
{
public:
	Environment();
	Environment(const std::string& model_meta_file, bool enable_key_worm);
	~Environment();
	void ParseArgs(const std::vector<std::string>& args);

	FEM::World* GetSoftWorld() {return mSoftWorld;};
	Worm* GetCreature() {return mCreature;};

    double GetSimulationHz() { return mSimulationHz; };
    double GetControlHz() { return mControlHz; };
    PeridynoBridge* GetFluidBridge() { return mFluidBridge; };
    int GetNumIters() { return mCurrIters; };
	void SetMaxIters(int nIters) { mMaxIters = nIters; };
	int GetMaxIters() { return mMaxIters; };
	void SetSampleCount(int n) { mSampleCount = n; };

	void Step();
	void StepKeyWorm() { if (mKeyCreature != nullptr) mKeyCreature->Step(); };
	void Reset();

	const Eigen::VectorXd& GetStates();
	int GetStateSize() const;

	void InitializeActions();
	const Eigen::VectorXd& GetActions() {return mActions;};
	int GetActionSize() const { return mActions.size(); };
	void SetActions(const Eigen::VectorXd& actions);
	void SetActions();

	int GetMuscleActivationSize() { return mCreature->GetNumMuscleSegments(); };

	std::map<std::string, double> CalcRewardImitate() const;
	std::map<std::string, double> GetRewards();

	bool isEndOfEpisode();
	bool CheckValidEpisode();

	Eigen::VectorXd GetNormLowerBound() {return mNormLowerBound;};
	Eigen::VectorXd GetNormUpperBound() {return mNormUpperBound;};
	Normalizer*	GetNormalizer() {return mNormalizer;};

	void SetPhase(const unsigned int& phase); 
	const unsigned int& GetPhase() { return mPhase; };

	Eigen::Vector3d GetAverageVelocity() {return mAverageVelocity;};
	Eigen::Vector3d GetAverageEvalCoordVelocity() { return mAverageEvalCoordVelocity; };
	Eigen::Vector3d GetCurrentCoordVelocity() { return mCurrentCoordVelocity; }
	Eigen::Vector3d GetTargetVelocity() {return mTargetVelocity;};
	Eigen::Vector3d GetAverageSamplingVelocity() { return mAverageSamplingVelocity; };
	Eigen::Affine3d GetEvalLocalCoord() { return mEvalLocalCoord; };
	
	void UpdateRandomTargetVelocity();
	void UpdateAverageVelocity();

	void RecordSimWormSignal();
	void WriteActions(const std::string& path_file, const Eigen::MatrixXd& acts);	
	void WriteStates(const std::string& path_file, const Eigen::MatrixXd& states);
	void WriteMuscleSignals(const std::string& path_file, const Eigen::MatrixXd& signals);

	Eigen::Vector3d UpdateHeadLocation();
	Eigen::Vector3d GetHeadLocation() { return mHeadLocation; }

	void SaveWormStates(const std::string& path) { mCreature->WriteMesh(path, mSoftWorld); }
	void RestoreWormStates(const std::string& path) { mCreature->SetWormStates(path, mSoftWorld); }

	// 设置体积/拓扑日志输出间隔（仿真步数，默认 1 秒）
	void SetVolumeLogIntervalSteps(int steps) { mVolumeLogIntervalSteps = steps; }
	int GetVolumeLogIntervalSteps() const { return mVolumeLogIntervalSteps; }

private:
	double CalcPosErr(Eigen::Vector3d p1, Eigen::Vector3d p2);
	double CalcVelErr(const Eigen::Vector3d v1, const Eigen::Vector3d& v2);
	void LogVolumeTopologyDiagnostics(double sph_ms, double fem_ms, double step_ms);

	// const Eigen::VectorXd& GetPos();		// todo: better in Worm?
	// const Eigen::VectorXd& GetVel();
	// const Eigen::Vector3d& GetCoordVel();

private:
	FEM::World*						mSoftWorld;
	Worm*							mCreature;
	KeyWorm*						mKeyCreature;
	Eigen::AlignedBox3d				mWormAABB;

	int 							mSimulationHz;
	int 							mControlHz;
	int                             mRecordLength;
	int 							mRecordCounter;
	int 							mCurrIters;
	int 							mMaxIters;
	int                             mSampleCount;

	Normalizer*						mNormalizer;
	Eigen::VectorXd 				mNormLowerBound;
	Eigen::VectorXd 				mNormUpperBound;

	// States
	Eigen::VectorXd					mStates;
	Eigen::VectorXd					mActions;
	int 							mStateSize;
	int 							mActionSize;

	unsigned int 					mPhase;
	Eigen::Affine3d 				mEvalLocalCoord;

	Eigen::Vector3d					mTargetVelocity;
	Eigen::Vector3d					mAverageSamplingVelocity;

	Eigen::Vector3d					mAverageVelocity;
	std::deque<Eigen::Vector3d>		mAverageVelocityDeque;

	Eigen::Vector3d 				mCurrentCoordVelocity;
	Eigen::Vector3d					mAverageEvalCoordVelocity;
	std::deque<Eigen::Vector3d>		mAverageEvalCoordPosDeque;

	std::shared_ptr<cArgParser> 	mArgParser;
	bool 							mbOutputFiles;

	// Json buffer
	Eigen::MatrixXd					mJsonActions;
	Eigen::MatrixXd					mJsonStates;
	Eigen::MatrixXd					mJsonMuscleActivations;

	Eigen::Vector3d mLocalCoordPosition;
	Eigen::Vector3d mHeadLocation;

	// 流体求解器桥接 (使用 Peridyno SPH 替代经验阻力公式)
	PeridynoBridge*					mFluidBridge;

	// 体积/拓扑诊断日志间隔（步数），默认 = mSimulationHz → 每 1 仿真秒
	int								mVolumeLogIntervalSteps;
};
#endif