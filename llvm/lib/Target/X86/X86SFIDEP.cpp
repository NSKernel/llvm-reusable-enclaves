/*
  The SGX-WASM Project written by NSKernel
  The Ohio State University

  Code inspired and modified from SGX-Shield

  X86ShepherdedMAccess.cpp
*/

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCContext.h"

#include <iostream>

using namespace llvm;

namespace {

class X86SFIDEP : public MachineFunctionPass {
public:
	static char ID;
	X86SFIDEP() : MachineFunctionPass(ID) {}

	StringRef getPassName() const override { return "X86 Shepherded Memory Access"; }

	bool runOnMachineFunction(MachineFunction &MF) override;
private:
	bool sanitize(MachineBasicBlock &MBB);

	bool alignBranch(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi);
	bool nforceRegisterLargerThanR15(unsigned reg, MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi, DebugLoc& DL);
	bool sanitizeRSP(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi, bool *codeInjected);
	//bool sanitizeRBP(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi);
	bool sanitizeMAccess(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi);

        void enforceRegisterLargerThanR15(unsigned reg, MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi, DebugLoc& DL);
	// For the use of recording the last added bundle
	MachineBasicBlock::iterator lastRFlagsChangedPointer;
	MachineBasicBlock::iterator lastBundlePointer;
	MachineBasicBlock *lastRFlagsChangedBB;

	const TargetInstrInfo *instrInfo;
	const TargetRegisterInfo *registerInfo;
};
char X86SFIDEP::ID = 0;

} // end anonymous namespace

bool X86SFIDEP::runOnMachineFunction(MachineFunction &MF) {
	bool modified = false;
	lastRFlagsChangedPointer = NULL;
	lastRFlagsChangedBB = NULL;

	if (MF.hasInlineAsm()) {
		// We do not handle inline ASM
		printf("X86SFIDEP: Warning: Inline ASM detected. This function is passed. Function name: ");
		std::cout << MF.getName().str() << std::endl;
		return false;
	}

	if (MF.getName().startswith("__cxx") || MF.getName().startswith("_GLOBAL")) {
		printf("X86SFIDEP: Possible C++ initialization thing. Ignored. Name = ");
		std::cout << MF.getName().str() << std::endl;
		return false;
	}
	// Setup target related metadata
	instrInfo = MF.getSubtarget().getInstrInfo();
	registerInfo = MF.getSubtarget().getRegisterInfo();

	// We allow unsanitized functions to do minimal initialization tasks
	// Such function shall start with a prefix of "__unsan_"
	if (MF.getName().startswith("__unsan_")) {
		//	(strncmp(MF.getName(), "__unsan_", 8))) {
		// Allow unsanitized functions
		/*
		if (MF.getName().startswith("__unsan_ucf_")) {
			// Critical functions. Add a ud2
			MachineFunction::iterator i = MF.begin();
			MachineBasicBlock &MBB = *i;
			MachineBasicBlock::iterator MBBi = MBB.begin();
			MachineInstr &MI = *MBBi;
			DebugLoc DL = MI.getDebugLoc();
			BuildMI(MBB, MBBi, DL, instrInfo->get(X86::TRAP));
			return true;
		}
		*/
		return false;
	}

	if (!MF.getName().compare(StringRef("main"))) {
		printf("X86SFIDEP: main found. Appended %%r15 = 0\n");
		MachineFunction::iterator MBB = MF.begin();
		MachineBasicBlock::iterator MBBI = MBB->begin();
		MachineInstr &MI = *MBBI;
		DebugLoc DL = MI.getDebugLoc();
		
		BuildMI(*MBB, MBBI, DL, instrInfo->get(X86::MOV64ri), X86::R15).addImm(0);
		modified = true;
	}

	for (MachineFunction::iterator i = MF.begin(), end = MF.end(); i != end; ++i) {
		modified |= sanitize(*i);	
	}

	return modified;
}

// Borrowed code from SGX-Shield
static bool hasControlFlow(const MachineInstr &MI) {
	return MI.getDesc().isBranch() ||
		MI.getDesc().isCall() ||
		MI.getDesc().isReturn() ||
		MI.getDesc().isTerminator() ||
		MI.getDesc().isBarrier(); // Jaebaek: we do not support barrier SFI
}

// False-positive-prone code. Could lead to performance issue
// but will not introduce false-negative
static bool effectOnRFlags(const MachineInstr &MI, const TargetRegisterInfo *registerInfo) {
	if (MI.modifiesRegister(X86::EFLAGS, registerInfo))
		return true;
	
	if (MI.getDesc().isCompare()) return true;
	if (hasControlFlow(MI)) return false;
	unsigned Opc = MI.getOpcode();
	if (X86::MOV16ao16 <= Opc && Opc <= X86::MOVZX64rr8) return false;
	if (X86::LEA16r <= Opc && Opc <= X86::LEA64r) return false;
	if (X86::CMOV16rr <= Opc && Opc <= X86::CMOV64rr) return false;
	if (X86::CMOV16rm <= Opc && Opc <= X86::CMOV64rm) return false;
	if (X86::CMOV_GR8 <= Opc && Opc <= X86::CMOV_VK64) return false;
	//if (X86::CMOVA16rm <= Opc && Opc <= X86::CMOV_V8I64) return false;
	if (Opc == X86::SETCCm && Opc == X86::SETCCr) return false;
	if (Opc == X86::LOOPE || Opc == X86::LOOPNE) return false;
	/*
	if (X86::DEC16m <= Opc && Opc <= X86::DEC8r) return true;
	*/
	return true;
}

static bool isCondInstr(const MachineInstr &MI) {
	if (MI.getDesc().isConditionalBranch()) return true;
	unsigned Opc = MI.getOpcode();
	//if (X86::CMOVA16rm <= Opc && Opc <= X86::CMOV_V8I64) return true;
	//if (X86::SETAEm <= Opc && Opc <= X86::SETSr) return true;
	if (X86::CMOV16rr <= Opc && Opc <= X86::CMOV64rr) return true;
	if (X86::CMOV16rm <= Opc && Opc <= X86::CMOV64rm) return true;
        if (X86::CMOV_GR8 <= Opc && Opc <= X86::CMOV_VK64) return true;
        //if (X86::CMOVA16rm <= Opc && Opc <= X86::CMOV_V8I64) return true;
        if (Opc == X86::SETCCm && Opc == X86::SETCCr) return true;
	if (Opc == X86::LOOPE || Opc == X86::LOOPNE) return true;
	return false;
}

static bool isDirectBranching(const MachineInstr &MI) {
	return MI.getDesc().isBranch() && !MI.getDesc().isIndirectBranch();
}

bool X86SFIDEP::alignBranch(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi) {
	// Note that we do not allow 16/32-bit branching via registers
	// This is not a big deal because if you generate 64-bit code,
	// you should not actually run into a 16/32-bit branching

	MachineInstr &MI = *MBBi;

	if (!hasControlFlow(MI)) {
		// Not branching
		return false;
	}

	if (isDirectBranching(MI)) {
		return false;
	}

	DebugLoc DL = MI.getDebugLoc();
	unsigned int Opcode = MI.getOpcode();
	
	// Get rid of 16-bit and 32-bit branching
	if (Opcode == X86::JMP16r || Opcode == X86::JMP32r ||
	    Opcode == X86::CALL16r || Opcode == X86::CALL32r ||
	    Opcode == X86::TAILJMPr) {
		// Should not be here
		llvm_unreachable("Jump target is not 64-bit. This is not expected.");
	}
	// Return. Replace return into pop r + jmp r
	// Needs to fall through to the branch sanitization
	if (Opcode == X86::RETW) {
		// Cannot be in 64-bit mode
		// Should not be here
		llvm_unreachable("Return is encoded as 16-bit. This is not expected.");
	}
	// We don't handle RETIX and LRETX because 
	// o SGX app cannot be an interrupt handler
	// o We only allow flat memory space so there won't be a far call
	else if (Opcode == X86::RETL) {
		// No sanitization for main
		if (!MBB.getParent()->getName().compare(StringRef("main"))) {
			printf("X86SFIDEP: main's return is left untouched\n");
			return false;
		}
		// Pop 32-bit to R14
		MachineInstr *popMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::POP32r)).addReg(X86::R14D);
		MachineBasicBlock::iterator POPi = popMI;

		// Align
		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::AND32ri8), X86::R14D).addReg(X86::R14D).addImm(-32);
		// To next basic block
		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::ADD32ri8), X86::R14D).addReg(X86::R14D).addImm(32);
		// RETL
		MachineInstr *jmpMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::JMP32r)).addReg(X86::R14D);
		MachineBasicBlock::iterator JMPI = jmpMI;

		lastBundlePointer = POPi;

		MI.eraseFromParent();
		MIBundleBuilder(MBB, POPi, ++JMPI);
		finalizeBundle(MBB, POPi.getInstrIterator());
		
		return true;
	}
	else if (Opcode == X86::RETQ) {
		// No sanitization for main
                if (!MBB.getParent()->getName().compare(StringRef("main"))) {
                        printf("X86SFIDEP: main's return is left untouched\n");
			return false;
                }
		// Pop 64-bit to R14
		MachineInstr *popMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::POP64r)).addReg(X86::R14);
		MachineBasicBlock::iterator POPi = popMI;

		// Align
		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::AND64ri8), X86::R14).addReg(X86::R14).addImm(-32);
		// To next basic block
		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::ADD64ri8), X86::R14).addReg(X86::R14).addImm(32);
		// RETL
		MachineInstr *jmpMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::JMP64r)).addReg(X86::R14);
		MachineBasicBlock::iterator JMPI = jmpMI;

		lastBundlePointer = POPi;

		MI.eraseFromParent();
		MIBundleBuilder(MBB, POPi, std::next(JMPI));
		finalizeBundle(MBB, POPi.getInstrIterator());

		return true;
	}


	// 64-bit branching. Do sanitization
	if (Opcode == X86::JMP64r ||
	    Opcode == X86::CALL64r ||
	    Opcode == X86::TAILJMPr64) {
		// We must ensure that all targets are aligned to 32 bytes
		unsigned RegisterX = MI.getOperand(0).getReg();
		
		MachineInstr *andMi = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::AND64ri8), RegisterX).addReg(RegisterX).addImm(-32);
		MachineBasicBlock::iterator ANDi = andMi;

		lastBundlePointer = ANDi;

		// Make our injected code plus the original branch instruction as a bundle
		MIBundleBuilder(MBB, ANDi, std::next(MBBi));
		finalizeBundle(MBB, ANDi.getInstrIterator());
		return true;
	}
	else if (Opcode == X86::JMP64m ||
	         Opcode == X86::CALL64m ||
	         Opcode == X86::TAILJMPm64) {
		// We must ensure that all targets are aligned to 32 bytes
		MachineOperand &BaseReg  = MI.getOperand(0);
		MachineOperand &Scale = MI.getOperand(1);
		MachineOperand &IndexReg  = MI.getOperand(2);
		MachineOperand &Disp = MI.getOperand(3);
		MachineOperand &SegmentReg = MI.getOperand(4);

		MachineInstr *movMi = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::MOV64rm)).addReg(X86::R14).addReg(BaseReg.getReg()).add(Scale).addReg(IndexReg.getReg()).add(Disp).addReg(0);
		
		MachineInstr *andMi = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::AND64ri8), X86::R14).addReg(X86::R14).addImm(-32);
		unsigned int newOpcode;

		if (Opcode == X86::JMP64m) {
			newOpcode = X86::JMP64r;
		}
		else if (Opcode == X86::CALL64m) {
			newOpcode = X86::CALL64r;
		}
		else if (Opcode == X86::TAILJMPm64) {
			newOpcode = X86::TAILJMPr64;
		}
		MachineInstr *callMi = BuildMI(MBB, MBBi, DL, instrInfo->get(newOpcode), X86::R14);
		MachineBasicBlock::iterator MOVi = movMi;
		MachineBasicBlock::iterator CALLi = callMi;

		lastBundlePointer = MOVi;

		// Make our injected code plus the original branch instruction as a bundle
		MI.eraseFromParent();
		MIBundleBuilder(MBB, MOVi, ++CALLi);
		finalizeBundle(MBB, MOVi.getInstrIterator());
		return true;
	}

	return false;
}

void X86SFIDEP::enforceRegisterLargerThanR15(unsigned reg, MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi, DebugLoc& DL) {
	// Inserts the following
	//    sub %r15, %rxx
	//    shl $1, %rxx
	//    shr $1, %rxx
	//    lea (%rxx, %r15, 1), %rxx
	// This sanitization can poop because it only allows %rxx - %r15 < 0xFFFFFFFF
	// Also this sanitization isn't even totally correct since it also requires
	// r15 < 0xFFFFFFFF00000001
	// But first an SGX app cannot be larger than 4GB, and an SGX app also cannot
	// reside in the kernel space. So neither of these will be a problem
	// Sum it up:
	// o Bad but safe behaviour: %rxx - %15 >= 4GB
	// o Unsafe: %r15 >= 0xFFFFFFFF000000001 (in kernel space)
	
	MachineFunction &MF = *MBB.getParent();

	MachineInstr *LEA = MF.CreateMachineInstr(instrInfo->get(X86::LEA64r), DL);
	MBB.insertAfter(MBBi, LEA);
	MachineInstrBuilder(MF, LEA).addReg(reg).addReg(reg).addImm(1).addReg(X86::R15).addImm(0).addReg(0);
	MachineBasicBlock::iterator LEAi = LEA;

	BuildMI(MBB, LEAi, DL, instrInfo->get(X86::SUB64rr), reg).addReg(reg).addReg(X86::R15);
	BuildMI(MBB, LEAi, DL, instrInfo->get(X86::SHL64ri), reg).addReg(reg).addImm(1);
        BuildMI(MBB, LEAi, DL, instrInfo->get(X86::SHR64ri), reg).addReg(reg).addImm(1);
	lastBundlePointer = MBBi;

	MIBundleBuilder(MBB, MBBi, ++LEAi);
	finalizeBundle(MBB, MBBi.getInstrIterator());
}

bool X86SFIDEP::sanitizeRSP(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi, bool *codeInjected) {
	// RSP may get fucked by tons of instructions from sub to mov
	// However most of them will not be generated by the compiler
	// so we only have to worry about those who will be:
	// o Substracted by a large number (when allocating new stack frame)
	//   Note that there will never be sub %rxx, %rsp
	// o Bitwise and - Who would do that?
	// - Added by a "large" number
	//   Note that when adding an immediate, it can only be as large as a
	//   32-bit integer. So no overflow is worried. We will sanitize out
	//   add %rxx, %rsp 
	// - mov xxx, %rsp (and 32-bit mov %ebp, %esp)
	//   We have RBP preserved so RBP is always safe. No sanitization for this
	//   But for other registers, we will have to do something
	// o leaq (addr), %rsp
	//   Let it done then sanitize

	MachineInstr &MI = *MBBi;

	if (!MI.modifiesRegister(X86::ESP, registerInfo)) {
		// Not even touching RSP
		return false;
	}

	if (MI.getDesc().isCall()) {
		// Call is nothing more than push
		return false;
	}

	DebugLoc DL = MI.getDebugLoc();
	unsigned int Opcode = MI.getOpcode();

	//if (Opcode == X86::PUSH64r || Opcode == X86::POP64r || Opcode == X86::PUSHF16 || Opcode == X86) {
		// Only push and pop. The change to RSP is controllable and will not
		// be over 8 bytes
	//	return false;
	//}

	switch (Opcode) {
	// PUSH/POP
	case X86::PUSH64r:
	case X86::PUSH64i8:
	case X86::PUSH64i32:
	case X86::PUSH64rmm:
	case X86::PUSH64rmr:	
	case X86::POP64r:
	case X86::PUSHF16:
	case X86::POPF16:
	case X86::PUSHF32:
        case X86::POPF32:
	case X86::PUSHF64:
        case X86::POPF64:
		return false;
	// SUB
	case X86::SUB64ri8:
	case X86::SUB64ri32:
		{
		// subq $a, %rsp
		// =>
		// subq %r15, %rsp   // Offset
		// subq $a, %rsp     // Offset - a
		// andq 0x7fffffffffffffff, %rsp
		//                   // Make sure that Offset is smaller
		//                   // than the one that can overflow
		//                   // since R15 must be smaller than 
		//                   // 0x8000000000000000
		// leaq (%rsp, %r15, 1), %rsp
		// unsigned sanitizeOpcode;

		MachineInstr *subMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::SUB64rr), X86::RSP).addReg(X86::RSP).addReg(X86::R15);
		MachineBasicBlock::iterator SUBi = subMI;

		BuildMI(MBB, MBBi, DL, instrInfo->get(Opcode), X86::RSP).addReg(X86::RSP).addImm(MI.getOperand(2).getImm());

		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::MOV64ri), X86::R14).addImm(0x7fffffffffffffff);
		BuildMI(MBB, MBBi, DL, instrInfo->get(X86::AND64rr), X86::RSP).addReg(X86::RSP).addReg(X86::R14);

		MachineInstr *leaMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::LEA64r)).addReg(X86::RSP).addReg(X86::RSP).addImm(1).addReg(X86::R15).addImm(0).addReg(0);
		MachineBasicBlock::iterator LEAi = leaMI;

		// Replace it
		MI.eraseFromParent();

		lastBundlePointer = SUBi;

		// Make it a bundle
		MIBundleBuilder(MBB, SUBi, ++LEAi);
		finalizeBundle(MBB, SUBi.getInstrIterator());
		return true;
		}
	case X86::SUB64rr: case X86::SUB32rr: case X86::SUB16rr:
	case X86::SUB8rr: case X86::SUB32ri: case X86::SUB32ri8:
	case X86::SUB16ri: case X86::SUB16ri8: case X86::SUB8ri:
		llvm_unreachable("Unexpected to see a SUBrr or SUB32 or SUB16 or SUB8 on RSP");

/* From SGX-Shield's code. Not making sense to me
   Who would ever do AND on RSP?
	// AND
	case X86::AND64ri8:
	case X86::AND64ri32:
		addedBundleHeader = MBBi;
		unsigned tmpOpc = (Opc == X86::AND64ri8) ? X86::AND32ri8 : X86::AND32ri;
		MI.setDesc(instrInfo->get(tmpOpc));
		MI.getOperand(0).setReg(X86::ESP);
		MI.getOperand(1).setReg(X86::ESP);
		resetRXBasedOnRZP(X86::RSP, MBB, MBBi, instrInfo, DL);
		return true;
	case X86::AND64rr: case X86::AND32rr: case X86::AND16rr:
	case X86::AND8rr: case X86::AND32ri: case X86::AND32ri8:
	case X86::AND16ri: case X86::AND16ri8: case X86::AND8ri:
		llvm_unreachable("Unexpected to see a ANDrr or AND32 or AND16 or AND8 on RSP");
*/

	// ADD
	case X86::ADD64ri8: 
	case X86::ADD64ri32:
		// Safe because it will not overflow in the user space
		// If RSP is too big then it could overflow,
		// but it requires it's address to be 0xFFFFFFFFXXXXXXXX
		// which is in the kernel space
		return false;
	case X86::ADD64rr: case X86::ADD32rr: case X86::ADD16rr:
	case X86::ADD8rr: case X86::ADD32ri: case X86::ADD32ri8:
	case X86::ADD16ri: case X86::ADD16ri8: case X86::ADD8ri:
		llvm_unreachable("Unexpected to see a ADDrr or ADD32 or ADD16 or ADD8 on RSP");

	// MOV
	case X86::MOV32rr:
		// Check which register
		if (MI.getOperand(1).getReg() == X86::EBP) {
			// We will promote this to a 64-bit move
			MI.getOperand(0).setReg(X86::RSP);
			MI.getOperand(1).setReg(X86::RBP);
			MI.setDesc(instrInfo->get(X86::MOV64rr));
			Opcode = X86::MOV64rr;
			// Modified but safe so no other code injected
			*codeInjected = false;
			return true;
		}
		else {
			// Other registers
			// Just do sanitization...
			// The lastBundlePointer is updated inside the following function
			enforceRegisterLargerThanR15(X86::RSP, MBB, MBBi, DL);
			return true;
		}
	case X86::MOV64rr:
		if (MI.getOperand(1).getReg() == X86::RBP) {
			// This is safe
			return false;
		}
		else {
			// Other registers
			// Just do sanitization...
			// The lastBundlePointer is updated inside the following function
			enforceRegisterLargerThanR15(X86::RSP, MBB, MBBi, DL);
			return true;
		}
	case X86::MOV32rm:
	case X86::MOV64rm:
		// Should we also sanitize the memory access...?
		enforceRegisterLargerThanR15(X86::RSP, MBB, MBBi, DL);
		return true;
	
	// LEA
	case X86::LEA32r:
		// Should not be here
		llvm_unreachable("Unexpected to see a LEA32r on RSP");
	case X86::LEA64_32r:
	case X86::LEA64r:
		// In fact lea (%rbp, offset, 1), %rsp is safe when offset is small
		// However to be simple, we just check everytime
		enforceRegisterLargerThanR15(X86::RSP, MBB, MBBi, DL);
		return true;
	}

	//llvm_unreachable
	printf("X86SFIDEP: Unexpected to see an unhandled instruction modifying RSP in");
	std::cout << MBB.getParent()->getName().str() << ". Instruction is: " << MI.getOpcode() << std::endl;
	return false;
}

/* We reserved RBP so this is not necessary anymore
bool X86SFIDEP::sanitizeRBP(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi) {
	MachineInstr &MI = *MBBi;


	return false;
}
*/

static bool findMemoryOperand(const MachineInstr &MI,
	SmallVectorImpl<unsigned>* indices) {
	int NumFound = 0;
	Register OperandReg;
	bool warnForUnsan = false;
	for (unsigned i = 0; i < MI.getNumOperands(); ) {
		if (MI.getOperand(i).isReg()) {
			// NOT SECURE! SHOULD USE ALTERNATIVE WAY
			// Go see the sanitization code's comment
			OperandReg = MI.getOperand(i).getReg();
			if (OperandReg == X86::AH || OperandReg == X86::CH ||
			    OperandReg == X86::BH || OperandReg == X86::DH) {
				warnForUnsan = true;
			}
		}
		if (isMem(MI, i)) {
			NumFound++;
			indices->push_back(i);
			i += X86::AddrNumOperands;
		}
		else {
			i++;
		}
	}

	// Intrinsics and other functions can have mayLoad and mayStore to reflect
	// the side effects of those functions.  This function is used to find
	// explicit memory references in the instruction, of which there are none.
	if (NumFound == 0)
		return false;
	if (warnForUnsan == true) {
		printf("X86SFIDEP: Warning! Unsanitized due to REX collision with high byte registers!\n!");
		return false;
	}
	return true;
}

bool X86SFIDEP::sanitizeMAccess(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBi) {
	MachineInstr &MI = *MBBi;
	unsigned Opcode = MI.getOpcode();

	if (!MI.mayStore()) // !MI.mayLoad() &&
		return false;

	if (Opcode == X86::PUSH64r || Opcode == X86::POP64r) {
		// Push and pop are safe since RSP is safe
		return false;
	}
		
	if (MI.getNumOperands() < 5 || MI.getOpcode() < X86::AAA)
		return false;
	
	DebugLoc DL = MI.getDebugLoc();
	SmallVector<unsigned, 2> MemOps;
	if (!findMemoryOperand(MI, &MemOps))
		return false;

	// Jaebaek: as a prototype, only support one memory operand
	assert(MemOps.size() <= 1);	
	for (unsigned MemOp : MemOps) {
		MachineOperand &BaseReg  = MI.getOperand(MemOp + 0);
		MachineOperand &Scale = MI.getOperand(MemOp + 1);
		MachineOperand &IndexReg  = MI.getOperand(MemOp + 2);
		MachineOperand &Disp = MI.getOperand(MemOp + 3);
		MachineOperand &SegmentReg = MI.getOperand(MemOp + 4);
		// If one of RIP, RBP, R12 and RSP is a base reg
		// and no index reg, it is safe
		// --> because the attacker cannot change the dest address
		if ((BaseReg.getReg() == X86::RIP
			|| BaseReg.getReg() == X86::RBP
			|| BaseReg.getReg() == X86::RSP
			|| BaseReg.getReg() == X86::R12)
			&& IndexReg.getReg() == 0
			&& SegmentReg.getReg() == 0)
		  	continue;	
		unsigned rX;
		MachineBasicBlock::iterator head;
		/*
		  before:
		   mov  src, (xx:base)
		  after:
		   sub  %r15, %base
		   mov  0x7fffffffffffffff, %r14
		   and  %
		   mov  src, (xx:%r15, %base, 1)
		 */
		if (0) {// (IndexReg.getReg() == 0) {
			rX = BaseReg.getReg();
			//unsigned rX32 = getX86SubSuperRegister(rX, 32, false);
			MachineInstr *subMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::SUB64rr), rX).addReg(rX).addReg(X86::R15);

			
			head = subMI;

		}
		else {
		  /*
		    before:
		     xxx  src, disp(base, index, scale)
		    after:
		     lea  disp(base, index, scale), %r14
		     sub  %r15, %r14
		     shl  $1, %r14
		     shr  $1, %r14
		     xxx  src, (%r15, %r14, 1)
		   */
		   /*
		    * TODO
		    * Note that as a workaround to REX-encoding
		    * preventing us from doing
		    *   xxx (%r15, %r14, 1), %ah
		    * We are temporarily disabling the sanitizaion
		    * for such instruction. We can use something
		    * like
		    *   backup rbx
		    *   lea (%r15, %r14, 1), %rbx
		    *   xxx src, (%rbx)
		    * but that is left to future works
		    */
			rX = X86::R14;
			MachineInstr *leaMI = BuildMI(MBB, MBBi, DL, instrInfo->get(X86::LEA64r)).addReg(X86::R14).addReg(BaseReg.getReg()).add(Scale).addReg(IndexReg.getReg()).add(Disp).addReg(0);
			head = leaMI;	
			BuildMI(MBB, MBBi, DL, instrInfo->get(X86::SUB64rr), X86::R14).addReg(X86::R14).addReg(X86::R15);
			BuildMI(MBB, MBBi, DL, instrInfo->get(X86::SHL64ri), X86::R14).addReg(X86::R14).addImm(1);
			BuildMI(MBB, MBBi, DL, instrInfo->get(X86::SHR64ri), X86::R14).addReg(X86::R14).addImm(1);
		}	
		MachineInstrBuilder rebuiltMI = BuildMI(MBB, MBBi, DL, MI.getDesc());
		for (unsigned i = 0;i < MI.getNumOperands();) {
			if (i != MemOp) {
				rebuiltMI.add(MI.getOperand(i));
				++i;
			} else {
				rebuiltMI.addReg(X86::R15).addImm(1).addReg(rX).addImm(0).addReg(SegmentReg.getReg());
				i += 5;
			}
		}
		MachineBasicBlock::iterator I = *rebuiltMI;

		MI.eraseFromParent();

		lastBundlePointer = head;
		MIBundleBuilder(MBB, head, ++I);
		finalizeBundle(MBB, head.getInstrIterator());
		return true;
	}
	return false;
}

bool X86SFIDEP::sanitize(MachineBasicBlock &MBB) {
	bool modified = false;

	// We do DEP on branching instructions and SFI boundary check on memory
	// accessses, however the code we inject might change RFLAGS
	// This leaves us to save RFLAGS and restore it after our injected code
	// We don't have to do this everytime. We save the last change of the flag
	// then restore it
	// The routine of the sanitize() starts by updating if current instruction
	// is changing the RFLAGS. Then it checks if we have to restore it because
	// of a condition instruction


	unsigned int i = 0, lastRFlagsChangedIndex = 0, lastInjectedCodeIndex = 0;
	for (MachineBasicBlock::iterator MBBi = MBB.begin(), NextMBBi = MBBi;
		MBBi != MBB.end(); MBBi = ++NextMBBi, ++i) {

		if (effectOnRFlags(*MBBi, registerInfo)) {
			// We save the *last RFLAGS modified* index and pointer
			lastRFlagsChangedIndex = i;
			lastRFlagsChangedPointer = MBBi;
			lastRFlagsChangedBB = &MBB;
		}

		if (isCondInstr(*MBBi)) {
			// We have to check if RFLAGS change by our injected code
			// will interfere with conditional instructions
			if (lastRFlagsChangedIndex < lastInjectedCodeIndex) {
				// We will save RFLAGS to R13 then restore it after the last
				// injected code

				// Save to R13
        			assert(lastRFlagsChangedPointer != NULL && lastRFlagsChangedBB != NULL);
				DebugLoc DL = (*lastRFlagsChangedPointer).getDebugLoc();
				MachineBasicBlock::iterator lastRFlagsChangedNext = std::next(lastRFlagsChangedPointer);	
				BuildMI(*lastRFlagsChangedBB, lastRFlagsChangedNext, DL, instrInfo->get(X86::PUSHF16));
				BuildMI(*lastRFlagsChangedBB, lastRFlagsChangedNext, DL, instrInfo->get(X86::POP16r)).addReg(X86::R13W);

				MachineBasicBlock::iterator lastBundleIterator = lastBundlePointer;
        			while (++lastBundleIterator != MBB.end() && (*lastBundleIterator).isInsideBundle());
        			DL = (*lastBundleIterator).getDebugLoc();
				// Now lastBundleIterator points to the first instruction out of the bundle
        			BuildMI(MBB, lastBundleIterator, DL, instrInfo->get(X86::PUSH16r)).addReg(X86::R13W);
				BuildMI(MBB, lastBundleIterator, DL, instrInfo->get(X86::POPF16));

        			lastInjectedCodeIndex = 0; // <-- this enforce pushf / popf only once for sub
			}
		}

		// Enforce DEP
		// We do not enforce DEP on direct branches (jmp $imm).
		// This is because the $imm is fused in the code and cannot be 
		// changed by a malicious attacker.
		// We only deal with those branch instructions whose target is 
		// stored in a register which a malicious attacker might load 
		// undesired value into
		bool codeInjected = true;
		if (alignBranch(MBB, MBBi)) {
			modified = true;
		}
		// Enforce SFI
		else if (sanitizeRSP(MBB, MBBi, &codeInjected) || /*sanitizeRBP(MBB, MBBi) ||*/ sanitizeMAccess(MBB, MBBi)) {
			if (codeInjected) {
				lastInjectedCodeIndex = i;
			}
			modified = true;
		}
		
	}

	return modified;
}


FunctionPass *llvm::createX86SFIDEP() { 
	return new X86SFIDEP(); 
}
