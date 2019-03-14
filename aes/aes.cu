#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "aes_cuda_table.h"

#define NUM_BLOCK_PER_MULTIPROCESSOR	3
#define SIZE_BLOCK_PER_MULTIPROCESSOR	256*1024
#define MAX_THREAD			64
#define STATE_THREAD		4
#define MAX_CHUNK_SIZE		(8*1024*1024)

#define AES_ENCRYPT		1
#define AES_DECRYPT		0
#define AES_MAXNR		14
#define AES_BLOCK_SIZE		16
#define AES_KEY_SIZE_128	16
#define AES_KEY_SIZE_192	24
#define AES_KEY_SIZE_256	32

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

#define ITERATION			5

#define CUDA_MRG_ERROR_CHECK(call) {																	\
		call;				                                												\
		cudaerrno=cudaGetLastError();																	\
		if(cudaSuccess!=cudaerrno) {                                       					         						\
			if (output_verbosity!=OUTPUT_QUIET) \
				fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",__FILE__,__LINE__,cudaGetErrorString(cudaerrno));	\
				exit(EXIT_FAILURE);                                                  											\
		} }

#define CUDA_MRG_ERROR_NOTIFY(msg) {                                    												\
		cudaerrno=cudaGetLastError();																	\
		if(cudaSuccess!=cudaerrno) {                                                											\
			if (output_verbosity!=OUTPUT_QUIET) \
				fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",__FILE__,__LINE__-3,cudaGetErrorString(cudaerrno));	\
				exit(EXIT_FAILURE);                                                  											\
		} }

typedef struct aes_key_st {
	unsigned int rd_key[4 *(AES_MAXNR + 1)];
	int rounds;
} AES_KEY;

static int output_verbosity;

uint32_t  *d_s;
uint8_t  *h_s;

int *d_key_round;
unsigned int *d_rd_key;

float elapsed;
cudaEvent_t start,stop;

void (*transferHostToDevice) (const unsigned char  **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size);
void (*transferDeviceToHost) (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size);

void transferHostToDevice_PINNED (const unsigned char **input, uint32_t **deviceMem, uint8_t **hostMem, size_t *size) {
	cudaError_t cudaerrno;
	memcpy(*hostMem,*input,*size);
	CUDA_MRG_ERROR_CHECK(cudaMemcpyAsync(*deviceMem, *hostMem, *size, cudaMemcpyHostToDevice, 0));
}

void transferDeviceToHost_PINNED   (unsigned char **output, uint32_t **deviceMem, uint8_t **hostMemS, uint8_t **hostMemOUT, size_t *size) {
	cudaError_t cudaerrno;
	CUDA_MRG_ERROR_CHECK(cudaMemcpyAsync(*hostMemS, *deviceMem, *size, cudaMemcpyDeviceToHost, 0));
	CUDA_MRG_ERROR_CHECK(cudaThreadSynchronize());
	memcpy(*output,*hostMemS,*size);
}

#define GETU32(p) (*((uint32_t*)(p)))

static const uint32_t rcon[] = {
		0x00000001U, 0x00000002U, 0x00000004U, 0x00000008U,
		0x00000010U, 0x00000020U, 0x00000040U, 0x00000080U,
		0x0000001bU, 0x00000036U, /* for 128-bit blocks, Rijndael never uses more than 10 rcon values */
};

/* Expand the cipher key into the encryption key schedule. */
int AES_cpu_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key) {
	uint32_t *rk;
	int i = 0;
	uint32_t temp;

	if (!userKey || !key) return -1;

	if (bits != 128 && bits != 192 && bits != 256) return -2;

	rk = key->rd_key;

	if (bits==128) key->rounds = 10;
	else if (bits==192) key->rounds = 12;
	else key->rounds = 14;

	rk[0] = GETU32(userKey     );
	rk[1] = GETU32(userKey +  4);
	rk[2] = GETU32(userKey +  8);
	rk[3] = GETU32(userKey + 12);
	if (bits == 128) {
		while (1) {
			temp  = rk[3];
			rk[4] = rk[0] ^ (Te4[(temp >>  8) & 0xff]      ) ^ (Te4[(temp >> 16) & 0xff] <<  8) ^
					(Te4[(temp >> 24)       ] << 16) ^ (Te4[(temp      ) & 0xff] << 24) ^ rcon[i];
			rk[5] = rk[1] ^ rk[4];
			rk[6] = rk[2] ^ rk[5];
			rk[7] = rk[3] ^ rk[6];
			if (++i == 10) return 0;
			rk += 4;
		}
	}
	rk[4] = GETU32(userKey + 16);
	rk[5] = GETU32(userKey + 20);
	if (bits == 192) {
		while (1) {
			temp  = rk[5];
			rk[6] = rk[0] ^ (Te4[(temp >>  8) & 0xff]      ) ^ (Te4[(temp >> 16) & 0xff] <<  8) ^
					(Te4[(temp >> 24)       ] << 16) ^ (Te4[(temp      ) & 0xff] << 24) ^ rcon[i];
			rk[7] = rk[1] ^ rk[6];
			rk[8] = rk[2] ^ rk[7];
			rk[9] = rk[3] ^ rk[8];
			if (++i == 8) return 0;
			rk[10] = rk[ 4] ^ rk[ 9];
			rk[11] = rk[ 5] ^ rk[10];
			rk += 6;
		}
	}
	rk[6] = GETU32(userKey + 24);
	rk[7] = GETU32(userKey + 28);
	if (bits == 256) {
		while (1) {
			temp = rk[7];
			rk[ 8] = rk[0] ^ (Te4[(temp >>  8) & 0xff]      ) ^ (Te4[(temp >> 16) & 0xff] <<  8) ^
					(Te4[(temp >> 24)       ] << 16) ^ (Te4[(temp      ) & 0xff] << 24) ^ rcon[i];
			rk[ 9] = rk[1] ^ rk[ 8];
			rk[10] = rk[2] ^ rk[ 9];
			rk[11] = rk[3] ^ rk[10];
			if (++i == 7) return 0;
			temp = rk[11];
			rk[12]=rk[ 4] ^ (Te4[(temp      ) & 0xff]      ) ^ (Te4[(temp >>  8) & 0xff] <<  8) ^
					(Te4[(temp >> 16) & 0xff] << 16) ^ (Te4[(temp >> 24)       ] << 24);
			rk[13]=rk[ 5] ^ rk[12];
			rk[14]=rk[ 6] ^ rk[13];
			rk[15]=rk[ 7] ^ rk[14];

			rk += 8;
		}
	}
	return 0;
}

__global__ void AESencKernel(uint32_t state[], unsigned int *d_rd_key, int *d_key_round) {

	__shared__ uint32_t t[MAX_THREAD];
	__shared__ uint32_t s[MAX_THREAD];
	int tidx = threadIdx.x, tidy = threadIdx.y;

	for (int i =0; i< ITERATION; i++) {
	s[tidx+4*tidy] = state[blockIdx.x*MAX_THREAD+tidx+4*tidy] ^ d_rd_key[tidx];

	/* round 1: */
	t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(s[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[4+tidx];
	/* round 2: */
	s[tidx+4*tidy] = Te0[t[tidx+4*tidy] & 0xff] ^ Te1[(t[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(t[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[t[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[8+tidx];
	/* round 3: */
	t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(s[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[12+tidx];
	/* round 4: */
	s[tidx+4*tidy] = Te0[t[tidx+4*tidy] & 0xff] ^ Te1[(t[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(t[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[t[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[16+tidx];
	/* round 5: */
	t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(s[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[20+tidx];
	/* round 6: */
	s[tidx+4*tidy] = Te0[t[tidx+4*tidy] & 0xff] ^ Te1[(t[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(t[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[t[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[24+tidx];
	/* round 7: */
	t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(s[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[28+tidx];
	/* round 8: */
	s[tidx+4*tidy] = Te0[t[tidx+4*tidy] & 0xff] ^ Te1[(t[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(t[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[t[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[32+tidx];
	/* round 9: */
	t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
			Te2[(s[(2+tidx)%4+4*tidy] >>  16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
			d_rd_key[36+tidx];
	if (*d_key_round > 10) {
		/* round 10: */
		s[tidx+4*tidy] = Te0[t[tidx+4*tidy] & 0xff] ^ Te1[(t[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
				Te2[(t[(2+tidx)%4+4*tidy] >> 16) & 0xff] ^ Te3[t[(3+tidx)%4+4*tidy] >> 24] ^
				d_rd_key[40+tidx];
		/* round 11: */
		t[tidx+4*tidy] = Te0[s[tidx+4*tidy] & 0xff] ^ Te1[(s[(1+tidx)%4+4*tidy] >> 8) & 0xff] ^
				Te2[(s[(2+tidx)%4+4*tidy] >> 16) & 0xff] ^ Te3[s[(3+tidx)%4+4*tidy] >> 24] ^
				d_rd_key[44+tidx];
		if (*d_key_round > 12) {
			/* round 12: */
			s[tidx+4*tidy] = Te0[ t[tidx        +4*tidy]        & 0xff] ^
					Te1[(t[(1+tidx)%4+4*tidy] >>  8) & 0xff] ^
					Te2[(t[(2+tidx)%4+4*tidy] >> 16) & 0xff] ^
					Te3[ t[(3+tidx)%4+4*tidy] >> 24        ] ^
					d_rd_key[48+tidx];
			/* round 13: */
			t[tidx+4*tidy] = Te0[ s[tidx        +4*tidy]        & 0xff] ^
					Te1[(s[(1+tidx)%4+4*tidy] >>  8) & 0xff] ^
					Te2[(s[(2+tidx)%4+4*tidy] >> 16) & 0xff] ^
					Te3[ s[(3+tidx)%4+4*tidy] >> 24        ] ^
					d_rd_key[52+tidx];
		}
	}
	/* last round: */
	s[tidx+4*tidy]= (Te2[(t[tidx+4*tidy]            ) & 0xff] & 0x000000ff) ^
			(Te3[(t[(1+tidx)%4+4*tidy] >>  8) & 0xff] & 0x0000ff00) ^
			(Te0[(t[(2+tidx)%4+4*tidy] >> 16) & 0xff] & 0x00ff0000) ^
			(Te1[(t[(3+tidx)%4+4*tidy] >> 24)       ] & 0xff000000) ^
			d_rd_key[tidx+(*d_key_round << 2)];

	}
	state[blockIdx.x*MAX_THREAD+tidx+4*tidy] = s[tidx+4*tidy];
}

/* Encrypt a single block in and out can overlap. */
extern "C" void AES_cuda_encrypt(const unsigned char *in, unsigned char *out, size_t nbytes) {
	if (output_verbosity==OUTPUT_VERBOSE) fprintf(stdout,"\nSize: %d\n",(int)nbytes);
	if (output_verbosity==OUTPUT_VERBOSE) fprintf(stdout,"Starting encrypt...");

	cudaError_t cudaerrno;

	assert(in && out && nbytes);

	transferHostToDevice (&in, &d_s, &h_s, &nbytes);

	if (output_verbosity==OUTPUT_VERBOSE) fprintf(stdout,"kernel execution...");
	if ((nbytes%(MAX_THREAD*STATE_THREAD))==0) {
		dim3 dimGrid(nbytes/(MAX_THREAD*STATE_THREAD));
		dim3 dimBlock(STATE_THREAD,MAX_THREAD/STATE_THREAD);
		CUDA_MRG_ERROR_CHECK(cudaEventRecord(start,0));
		AESencKernel<<<dimGrid,dimBlock>>>(d_s, d_rd_key, d_key_round);
		CUDA_MRG_ERROR_CHECK(cudaEventRecord(stop,0));
		CUDA_MRG_ERROR_CHECK(cudaEventSynchronize(stop));
		CUDA_MRG_ERROR_CHECK(cudaEventElapsedTime(&elapsed,start,stop));
		CUDA_MRG_ERROR_NOTIFY("kernel launch failure");
	}

	transferDeviceToHost (&out, &d_s, &h_s, &h_s, &nbytes);

	if (output_verbosity==OUTPUT_VERBOSE) fprintf(stdout,"done!\n");
}

void AES_cuda_transfer_key(const AES_KEY *key) {
	assert(key);
	cudaError_t cudaerrno;
	CUDA_MRG_ERROR_CHECK(cudaMemcpy(d_key_round, &(key->rounds), sizeof(int), cudaMemcpyHostToDevice));
	CUDA_MRG_ERROR_CHECK(cudaMemcpy(d_rd_key, &(key->rd_key), 4*(AES_MAXNR+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

extern "C" int AES_cuda_set_encrypt_key(unsigned char *userKey, int bits, AES_KEY *key){
	if(AES_cpu_set_encrypt_key(userKey,bits,key)!=0) return 1;
	AES_cuda_transfer_key(key);
	return 0;
}

extern "C" void AES_cuda_finish() {
	cudaError_t cudaerrno;

	CUDA_MRG_ERROR_CHECK(cudaFree(d_s));

	fprintf(stdout,"\nTotal time: %f milliseconds\n",elapsed);
	// free device & host key memory
	CUDA_MRG_ERROR_CHECK(cudaFree(d_key_round));
	CUDA_MRG_ERROR_CHECK(cudaFree(d_rd_key));
}

extern "C" void AES_cuda_init(int* nm,int buffer_size_engine,int output_kind) {
	assert(nm);
	cudaError_t cudaerrno;
	int deviceCount,buffer_size;
	cudaDeviceProp deviceProp;

	output_verbosity=output_kind;

	CUDA_MRG_ERROR_CHECK(cudaGetDeviceCount(&deviceCount));
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		if (output_verbosity!=OUTPUT_QUIET) 
			fprintf(stderr,"There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	} else {
		if (output_verbosity>OUTPUT_NORMAL)
			fprintf(stdout,"Successfully found a device supporting CUDA (CUDART_VERSION %d).\n",CUDART_VERSION);
	}
	CUDA_MRG_ERROR_CHECK(cudaSetDevice(0));
	CUDA_MRG_ERROR_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

	if (output_verbosity==OUTPUT_VERBOSE) {
		fprintf(stdout,"\nDevice %d: \"%s\"\n", 0, deviceProp.name);
		fprintf(stdout,"  SharedMemoryPerBlock:							 %zu\n", deviceProp.sharedMemPerBlock);
		fprintf(stdout,"  CUDA Capability Major revision number:         %d\n", deviceProp.major);
		fprintf(stdout,"  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
		fprintf(stdout,"  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		fprintf(stdout,"  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		fprintf(stdout,"  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		fprintf(stdout,"  Total global memory                            %fG\n", float(deviceProp.totalGlobalMem)/1024/1024/1024);
		fprintf(stdout,"  Warp size                                      %d\n", deviceProp.warpSize);
		fprintf(stdout,"  Max threads per block                          %d\n", deviceProp.maxThreadsPerBlock);
		fprintf(stdout,"  Max threads dim x                              %d\n", deviceProp.maxThreadsDim[0]);
		fprintf(stdout,"  Max threads dim y                              %d\n", deviceProp.maxThreadsDim[1]);
		fprintf(stdout,"  Max threads dim z                              %d\n", deviceProp.maxThreadsDim[2]);
		fprintf(stdout,"  Max block   dim x                              %d\n", deviceProp.maxGridSize[0]);
		fprintf(stdout,"  Max block   dim y                              %d\n", deviceProp.maxGridSize[1]);
		fprintf(stdout,"  Max block   dim z                              %d\n", deviceProp.maxGridSize[2]);
		fprintf(stdout,"\n");
	}

	if(buffer_size_engine==0)
		buffer_size=MAX_CHUNK_SIZE;
	else buffer_size=buffer_size_engine;

	*nm=deviceProp.multiProcessorCount;
	// set L1 cache and shared memory
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);


	//pinned memory mode - use special function to get OS-pinned memory
	CUDA_MRG_ERROR_CHECK(cudaHostAlloc( (void**)&h_s, buffer_size, cudaHostAllocDefault));
	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"Using pinned memory: cudaHostAllocDefault.\n");
	transferHostToDevice = transferHostToDevice_PINNED;	// set memory transfer function
	transferDeviceToHost = transferDeviceToHost_PINNED;	// set memory transfer function
	CUDA_MRG_ERROR_CHECK(cudaMalloc((void **)&d_s,buffer_size));

	if (output_verbosity!=OUTPUT_QUIET) fprintf(stdout,"The current buffer size is %d.\n\n", buffer_size);
	CUDA_MRG_ERROR_CHECK(cudaEventCreate(&start));
	CUDA_MRG_ERROR_CHECK(cudaEventCreate(&stop));


	//device key memory alloc
	CUDA_MRG_ERROR_CHECK(cudaMalloc((void **)&d_key_round, sizeof(int)));
	CUDA_MRG_ERROR_CHECK(cudaMalloc((void **)&d_rd_key, 4*(AES_MAXNR+1)*sizeof(unsigned int)));
}


char char2num(char input)
{
	if (input>=48 && input<=57)
		return input-48;
	else if (input>=65 && input<=70)
		return input-55;
	else if (input>=97 && input<=102)
		return input-87;
	return -1;
}

void get_random_text(unsigned char *text)
{
	for (int i=0; i<16; i++)
	{
		text[i]=(unsigned char)(rand()&0xff);
	}
}

int main(int argc, char *argv[])
{
	unsigned char usrKey[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
	unsigned char text[16]={0x40,0x32,0xAF,0x8D,0x61,0x03,0x51,0x23,0x90,0x6E,0x58,0xE0,0x67,0x14,0xFF,0xC5};
	unsigned char text1[16]={0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
	unsigned char text2[16]={0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
	unsigned char *in, *out;
	int bits = 128;
	AES_KEY key;
	int nm, buffer_size_engine = 0, verbosity = 1;
	int repeat=1, length=1, random=0;
	/* 0: no random;
	 * 1: random input between repeat, same for each run;
	 * 2: same input between repeat, random for each run;
	 * 3: random between repeat and each run;
	 * 4: chose text1[] as input;
	 * 5: chose text2[] as input;
	 * 6: obtain text from argv;
	 */
	if (argc>=2)
		repeat = atoi(argv[1]);
	if (argc>=3)
		length = atoi(argv[2]);
	if (argc>=4)
		random = atoi(argv[3]);
	if (argc>=5)
	{
		if (strlen(argv[4]) < 32)
		{
			printf ("size of argv[4] %d\n", (int)sizeof(argv[4]));
			printf("arg 4 should be longer than 32\n");
			return -1;
		}
		for (int i = 0; i < 16; i++)
		{
			text[i] = char2num(argv[4][2*i])*16 + char2num(argv[4][2*i+1]);
		}
	}
	// malloc for in out
	if (!(in = (unsigned char *)malloc(length*16)))
	{
		printf("Can't allocate memory for input.\n");
		exit(0);
	}
	if (!(out = (unsigned char *)malloc(length*16)))
	{
		printf("Can't allocate memory for output.\n");
		exit(0);
	}



	printf("number of repeat = %d\n", repeat);
	printf("number of block = %d\n", length);
	printf("random = %d\n\n", random);

	buffer_size_engine = 16*length;
	AES_cuda_init(&nm, buffer_size_engine, verbosity);

	AES_cuda_set_encrypt_key(usrKey, bits, &key);

	printf("usrkey\n");
	for(int i=0; i<16; i++)
	{
		printf("%02x", usrKey[i]);
	}
	printf("\n\n");

//-------------------enc begin------------------------------
	for (int i=0; i<repeat; i++)
	{
		if (random >= 2 && random <=5)
		{
			srand(time(NULL));
			get_random_text(text);
		}
		else
			srand(0);

		for(int i=0; i<length; i++)
		{
			switch (random)
			{
			case 1: case 3:
				get_random_text(text);
				memcpy(in+16*i, text, 16);
				break;
			case 4:
				memcpy(in+16*i, text1, 16);
				break;
			case 5:
				memcpy(in+16*i, text2, 16);
				break;
			default:
				memcpy(in+16*i, text, 16);
				break;
			}
		}
		AES_cuda_encrypt(in, out, 16*length);
	}
//-------------------enc end--------------------------------
	printf("enc_in first\n");
	for(int i=0; i<16; i++)
	{
		printf("%02x", in[i]);
	}
	printf("\n");

	printf("enc_in last\n");
	for(int i=0; i<16; i++)
	{
		printf("%02x", in[i+(length-1)*16]);
	}
	printf("\n\n");

	printf("enc_out first\n");
	for(int i=0; i<16; i++)
	{
		printf("%02x", out[i]);
	}
	printf("\n");

	printf("enc_out last\n");
	for(int i=0; i<16; i++)
	{
		printf("%02x", out[i+(length-1)*16]);
	}
	printf("\n");
	// Free memory of in and out
	free(in);
	free(out);
	AES_cuda_finish();
	return 0;
}
