package go_triton_client

import (
	"context"
	"github.com/okieraised/go-triton-client/triton_proto"
	"google.golang.org/grpc"
	"time"
)

type TritonGRPCClient struct {
	serverURL  string
	grpcConn   *grpc.ClientConn
	grpcClient triton_proto.GRPCInferenceServiceClient
}

// NewTritonGRPCClient inits a new gRPC client
func NewTritonGRPCClient(serverURL string, grpcOpts ...grpc.DialOption) (*TritonGRPCClient, error) {
	grpcConn, err := grpc.NewClient(serverURL, grpcOpts...)
	if err != nil {
		return nil, err
	}
	grpcClient := triton_proto.NewGRPCInferenceServiceClient(grpcConn)
	tritonGRPCClient := &TritonGRPCClient{
		serverURL:  serverURL,
		grpcConn:   grpcConn,
		grpcClient: grpcClient,
	}
	return tritonGRPCClient, nil
}

// ServerAlive check server is alive.
func (tc *TritonGRPCClient) ServerAlive(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverLiveResponse, err := tc.grpcClient.ServerLive(ctx, &triton_proto.ServerLiveRequest{})
	if err != nil {
		return false, err
	}
	return serverLiveResponse.Live, nil
}

// ServerReady check server is ready.
func (tc *TritonGRPCClient) ServerReady(timeout time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverReadyResponse, err := tc.grpcClient.ServerReady(ctx, &triton_proto.ServerReadyRequest{})
	if err != nil {
		return false, err
	}
	return serverReadyResponse.Ready, nil
}

// ServerMetadata Get server metadata.
func (tc *TritonGRPCClient) ServerMetadata(timeout time.Duration) (*triton_proto.ServerMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	serverMetadataResponse, err := tc.grpcClient.ServerMetadata(ctx, &triton_proto.ServerMetadataRequest{})
	return serverMetadataResponse, err
}

// ModelRepositoryIndex Get model repo index.
func (tc *TritonGRPCClient) ModelRepositoryIndex(timeout time.Duration, repoName string, isReady bool) (*triton_proto.RepositoryIndexResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	repositoryIndexResponse, err := tc.grpcClient.RepositoryIndex(ctx, &triton_proto.RepositoryIndexRequest{RepositoryName: repoName, Ready: isReady})
	return repositoryIndexResponse, err
}

// GetModelConfiguration Get model configuration.
func (tc *TritonGRPCClient) GetModelConfiguration(timeout time.Duration, modelName, modelVersion string) (*triton_proto.ModelConfigResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelConfigResponse, err := tc.grpcClient.ModelConfig(ctx, &triton_proto.ModelConfigRequest{Name: modelName, Version: modelVersion})
	return modelConfigResponse, err
}

// ModelInferStats Get Model infer stats.
func (tc *TritonGRPCClient) ModelInferStats(timeout time.Duration, modelName, modelVersion string) (*triton_proto.ModelStatisticsResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelStatisticsResponse, err := tc.grpcClient.ModelStatistics(ctx, &triton_proto.ModelStatisticsRequest{Name: modelName, Version: modelVersion})
	return modelStatisticsResponse, err
}

// ModelLoadWithGRPC Load Model with grpc.
func (tc *TritonGRPCClient) ModelLoadWithGRPC(timeout time.Duration, repoName, modelName string, modelConfigBody map[string]*triton_proto.ModelRepositoryParameter) (*triton_proto.RepositoryModelLoadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	loadResponse, err := tc.grpcClient.RepositoryModelLoad(ctx, &triton_proto.RepositoryModelLoadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return loadResponse, err
}

// ModelUnloadWithGRPC Unload model with grpc modelConfigBody if not is nil.
func (tc *TritonGRPCClient) ModelUnloadWithGRPC(timeout time.Duration, repoName, modelName string, modelConfigBody map[string]*triton_proto.ModelRepositoryParameter) (*triton_proto.RepositoryModelUnloadResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	unloadResponse, err := tc.grpcClient.RepositoryModelUnload(ctx, &triton_proto.RepositoryModelUnloadRequest{
		RepositoryName: repoName,
		ModelName:      modelName,
		Parameters:     modelConfigBody,
	})
	return unloadResponse, err
}

// ShareMemoryStatus Get share memory / cuda memory status.
func (tc *TritonGRPCClient) ShareMemoryStatus(timeout time.Duration, isCUDA bool, regionName string) (interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if isCUDA {
		cudaSharedMemoryStatusResponse, err := tc.grpcClient.CudaSharedMemoryStatus(ctx, &triton_proto.CudaSharedMemoryStatusRequest{Name: regionName})
		return cudaSharedMemoryStatusResponse, err
	}
	systemSharedMemoryStatusResponse, err := tc.grpcClient.SystemSharedMemoryStatus(ctx, &triton_proto.SystemSharedMemoryStatusRequest{Name: regionName})
	if err != nil {
		return nil, err
	}
	return systemSharedMemoryStatusResponse, nil
}

// ShareCUDAMemoryRegister cuda share memory register.
func (tc *TritonGRPCClient) ShareCUDAMemoryRegister(timeout time.Duration, regionName string, cudaRawHandle []byte, cudaDeviceID int64, byteSize uint64) (*triton_proto.CudaSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryRegisterResponse, err := tc.grpcClient.CudaSharedMemoryRegister(
		ctx, &triton_proto.CudaSharedMemoryRegisterRequest{
			Name:      regionName,
			RawHandle: cudaRawHandle,
			DeviceId:  cudaDeviceID,
			ByteSize:  byteSize,
		},
	)
	return cudaSharedMemoryRegisterResponse, err
}

// ShareCUDAMemoryUnRegister cuda share memory unregister.
func (tc *TritonGRPCClient) ShareCUDAMemoryUnRegister(timeout time.Duration, regionName string) (*triton_proto.CudaSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cudaSharedMemoryUnRegisterResponse, err := tc.grpcClient.CudaSharedMemoryUnregister(ctx, &triton_proto.CudaSharedMemoryUnregisterRequest{Name: regionName})
	return cudaSharedMemoryUnRegisterResponse, err
}

// ShareSystemMemoryRegister system share memory register.
func (tc *TritonGRPCClient) ShareSystemMemoryRegister(timeout time.Duration, regionName, cpuMemRegionKey string, byteSize, cpuMemOffset uint64) (*triton_proto.SystemSharedMemoryRegisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryRegisterResponse, err := tc.grpcClient.SystemSharedMemoryRegister(
		ctx, &triton_proto.SystemSharedMemoryRegisterRequest{
			Name:     regionName,
			Key:      cpuMemRegionKey,
			Offset:   cpuMemOffset,
			ByteSize: byteSize,
		},
	)
	return systemSharedMemoryRegisterResponse, err
}

// ShareSystemMemoryUnRegister system share memory unregister.
func (tc *TritonGRPCClient) ShareSystemMemoryUnRegister(timeout time.Duration, regionName string) (*triton_proto.SystemSharedMemoryUnregisterResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	systemSharedMemoryUnRegisterResponse, err := tc.grpcClient.SystemSharedMemoryUnregister(ctx, &triton_proto.SystemSharedMemoryUnregisterRequest{Name: regionName})
	return systemSharedMemoryUnRegisterResponse, err
}

// GetModelTracingSetting get model tracing setting.
func (tc *TritonGRPCClient) GetModelTracingSetting(timeout time.Duration, modelName string) (*triton_proto.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := tc.grpcClient.TraceSetting(ctx, &triton_proto.TraceSettingRequest{ModelName: modelName})
	return traceSettingResponse, err
}

// SetModelTracingSetting set model tracing setting.
func (tc *TritonGRPCClient) SetModelTracingSetting(timeout time.Duration, modelName string, settingMap map[string]*triton_proto.TraceSettingRequest_SettingValue) (*triton_proto.TraceSettingResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	traceSettingResponse, err := tc.grpcClient.TraceSetting(ctx, &triton_proto.TraceSettingRequest{ModelName: modelName, Settings: settingMap})
	return traceSettingResponse, err
}

func (tc *TritonGRPCClient) Disconnect() error {
	err := tc.grpcConn.Close()
	return err
}

// ModelGRPCInfer Call Triton with GRPC
func (tc *TritonGRPCClient) ModelGRPCInfer(timeout time.Duration, request *triton_proto.ModelInferRequest) (*triton_proto.ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	modelInferResponse, err := tc.grpcClient.ModelInfer(ctx, request)
	if err != nil {
		return nil, err
	}
	return modelInferResponse, nil
}
